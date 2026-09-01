import inspect
import os
import numpy as np
from PIL import Image, ImageFilter

from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm import tqdm
from transformers import CLIPProcessor

from models.ImageEncoder import ImageEncoder
from models.ReferenceNet import ReferenceNet

from pipeline import InferencePipeline
from diffusers.models import UNet2DConditionModel

from utils.util import save_videos_grid, preprocess, save_cube_lut, upsample_lut
from utils.videoreader import VideoReader

from accelerate.utils import set_seed
from einops import rearrange
from pillow_lut import identity_table
import time
class Inference():
    def __init__(self, config="configs/prompts/video_demo.yaml") -> None:
        print("Initializing LUT Generation Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)
        config  = OmegaConf.load(config)  
        inference_config = OmegaConf.load(config.inference_config)
        
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        weight_dtype = torch.float16 if device == 'cuda' else torch.float32

        ### >>> create diffusion pipeline >>> ###
        vae = AutoencoderKL.from_pretrained(config.pretrained_sd_path, subfolder="vae", torch_dtype=weight_dtype, low_cpu_mem_usage=True)
        self.clip_image_encoder = ImageEncoder(model_path=config.pretrained_clip_path).to(dtype=weight_dtype)
        self.clip_image_processor = CLIPProcessor.from_pretrained(config.pretrained_clip_path, local_files_only=True)

        unet = UNet2DConditionModel.from_pretrained(config.pretrained_sd_path, subfolder="unet", in_channels=6, out_channels=3, low_cpu_mem_usage=False, ignore_mismatched_sizes=True, torch_dtype=weight_dtype)
        state_dict = torch.load(config.pretrained_LD_path, map_location="cpu")['unet_state_dict']
        m, u = unet.load_state_dict(state_dict, strict=True)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)} ###")
        if len(m) !=0 or len(u) !=0:
            print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n ###")
        
        del state_dict
        import gc
        gc.collect()
        if device == 'mps': torch.mps.empty_cache()
        
        self.referencenet = ReferenceNet.from_pretrained(config.pretrained_sd_path, subfolder="unet", torch_dtype=weight_dtype, low_cpu_mem_usage=True)
        state_dict = torch.load(config.pretrained_GE_path, map_location="cpu")["referencenet_state_dict"]
        m, u = self.referencenet.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        if device == 'mps': torch.mps.empty_cache()
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)} ###")
        if len(m) !=0 or len(u) !=0:
            print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n ###")

        self.id_lut_hwc = identity_table(16).table.reshape(64, 64, 3)
        self.id_lut_chw = torch.from_numpy(rearrange(self.id_lut_hwc, "h w c -> c h w")).unsqueeze(0).to(dtype=weight_dtype, device=device)

        self.pipeline = InferencePipeline(
            vae=vae, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        )
        self.pipeline.to(device, torch_dtype=weight_dtype)
        try:
            self.pipeline.enable_attention_slicing()
        except Exception:
            pass
        print("Initialization Done!")

    def __call__(self, ref_sequence, input_path=None, save_path=None, random_seed=42, step=25, size=512, ncc=False, input_frames=None, return_frames=False, save_lut_path=None):
        if input_frames is not None:
            input_video = input_frames
        else:
            input_video = VideoReader(input_path).read()
        input_video, input_video_resize = preprocess(input_video, ref_sequence, size, ncc)

        random_seed = int(random_seed)
        if random_seed != -1: 
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()
        step = int(step)

        generator_device = "cpu" if getattr(self.pipeline.device, "type", str(self.pipeline.device)) == "mps" else self.pipeline.device
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(torch.initial_seed())
    
        lut = self.pipeline(
            num_inference_steps      = step,
            width                    = size,
            height                   = size,
            generator                = generator,
            num_actual_inference_steps = step,
            source_image             = ref_sequence,
            referencenet             = self.referencenet,
            clip_image_processor     = self.clip_image_processor,
            clip_image_encoder       = self.clip_image_encoder,
            input_video              = input_video_resize,
            id_lut                   = self.id_lut_chw,
            return_dict = False
        )
    
        lut = lut[0].detach().cpu().numpy()
        lut = rearrange(lut, "c h w -> h w c")
        lut = lut + self.id_lut_hwc
        lut = np.clip(lut, 0.0, 1.0)
        lut_flat = lut.flatten()
        if save_lut_path:
            save_cube_lut(lut_flat, save_lut_path, size=16)
        
        # Upsample the 16-point AI LUT to a massive 64-point LUT for high fidelity grading
        print("Upsampling LUT to 64x64x64 for high-quality grading...")
        lut_flat_64 = upsample_lut(lut_flat, old_size=16, new_size=64)
        lut = ImageFilter.Color3DLUT(64, lut_flat_64)
        output_frames = []

        for frame in tqdm(input_video):
            output_frame = np.array(Image.fromarray(frame).filter(lut))/ 255.0
            output_frames.append(output_frame)
            
        if return_frames:
            return [ (f * 255).astype(np.uint8) for f in output_frames ]

        if not save_path:
            return None

        output_frames = np.array(output_frames) 
        output_frames = rearrange(output_frames, "t h w c -> 1 c t h w")
        output_video = torch.from_numpy(output_frames)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        save_videos_grid(output_video, save_path)
        
        return save_path
