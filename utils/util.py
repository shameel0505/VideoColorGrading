# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Adapted from https://github.com/guoyww/AnimateDiff
import os
import imageio
import numpy as np

import torch
import torchvision

from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import torch.distributed as dist


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=25):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_images_grid(images: torch.Tensor, path: str):
    assert images.shape[2] == 1 # no time dimension
    images = images.squeeze(2)
    grid = torchvision.utils.make_grid(images)
    grid = (grid * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def video2images(path, step=4, length=16, start=0):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(np.array(frame))
    frames = frames[start::step][:length]
    return frames


def images2video(video, path, fps=8):
    imageio.mimsave(path, video, fps=fps)
    return


tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation

def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()


def vars(src, ref):

    r, z = src.reshape([-1, src.shape[-1]]).T, ref.reshape([-1, ref.shape[2]]).T

    cov_r, cov_z = np.cov(r), np.cov(z)

    mu_r, mu_z = r.mean(axis=1)[..., np.newaxis], z.mean(axis=1)[..., np.newaxis]
    eig_val_r, eig_vec_r = np.linalg.eig(cov_r)
    eig_val_r[eig_val_r < 0] = 0
    val_r = np.diag(np.sqrt(eig_val_r[::-1]))
    vec_r = np.array(eig_vec_r[:, ::-1])
    inv_r = np.diag(1. / (np.diag(val_r + np.spacing(1))))

    mat_c = val_r @ vec_r.T @ cov_z @ vec_r @ val_r
    eig_val_c, eig_vec_c = np.linalg.eig(mat_c)
    eig_val_c[eig_val_c < 0] = 0
    val_c = np.diag(np.sqrt(eig_val_c))

    transfer_mat = vec_r @ inv_r @ eig_vec_c @ val_c @ eig_vec_c.T @ inv_r @ vec_r.T
    return [cov_r, cov_z, mu_r, mu_z, transfer_mat]

def transfer(src, ref, variables):
    cov_r, cov_z, mu_r, mu_z, transfer_mat = variables
    r, z = src.reshape([-1, src.shape[2]]).T, ref.reshape([-1, ref.shape[2]]).T

    res = np.dot(transfer_mat, r - mu_r) + mu_z

    res = res.T.reshape(src.shape)

    return res

def preprocess(src, ref, size, ncc):
    input_video = [np.array(Image.fromarray(c)) for c in src][:480]
    if not ncc:
        output_frames = []
        input_video_cc = [np.array(Image.fromarray(c).resize((256, 256))) for c in input_video]
        variables = vars(np.array(input_video_cc), ref)
        for i, frame in enumerate(input_video):
            img_res = transfer(frame, ref, variables)
            img_res = np.clip(img_res, 0, 255).astype(np.uint8)
            output_frames.append(img_res)
        input_video = output_frames

    input_video_resize = [np.array(Image.fromarray(c).resize((size, size), Image.Resampling.LANCZOS)) for c in input_video]

    return input_video, input_video_resize

def save_cube_lut(lut_array, path, size=16, title="AI_Generated_LUT"):
    lut_array = np.array(lut_array).flatten()
    with open(path, 'w') as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f'LUT_3D_SIZE {size}\n')
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Adapted from https://github.com/guoyww/AnimateDiff
import os
import imageio
import numpy as np

import torch
import torchvision

from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import torch.distributed as dist


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=25):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_images_grid(images: torch.Tensor, path: str):
    assert images.shape[2] == 1 # no time dimension
    images = images.squeeze(2)
    grid = torchvision.utils.make_grid(images)
    grid = (grid * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def video2images(path, step=4, length=16, start=0):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(np.array(frame))
    frames = frames[start::step][:length]
    return frames


def images2video(video, path, fps=8):
    imageio.mimsave(path, video, fps=fps)
    return


tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation

def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()


def vars(src, ref):

    r, z = src.reshape([-1, src.shape[-1]]).T, ref.reshape([-1, ref.shape[2]]).T

    cov_r, cov_z = np.cov(r), np.cov(z)

    mu_r, mu_z = r.mean(axis=1)[..., np.newaxis], z.mean(axis=1)[..., np.newaxis]
    eig_val_r, eig_vec_r = np.linalg.eig(cov_r)
    eig_val_r[eig_val_r < 0] = 0
    val_r = np.diag(np.sqrt(eig_val_r[::-1]))
    vec_r = np.array(eig_vec_r[:, ::-1])
    inv_r = np.diag(1. / (np.diag(val_r + np.spacing(1))))

    mat_c = val_r @ vec_r.T @ cov_z @ vec_r @ val_r
    eig_val_c, eig_vec_c = np.linalg.eig(mat_c)
    eig_val_c[eig_val_c < 0] = 0
    val_c = np.diag(np.sqrt(eig_val_c))

    transfer_mat = vec_r @ inv_r @ eig_vec_c @ val_c @ eig_vec_c.T @ inv_r @ vec_r.T
    return [cov_r, cov_z, mu_r, mu_z, transfer_mat]

def transfer(src, ref, variables):
    cov_r, cov_z, mu_r, mu_z, transfer_mat = variables
    r, z = src.reshape([-1, src.shape[2]]).T, ref.reshape([-1, ref.shape[2]]).T

    res = np.dot(transfer_mat, r - mu_r) + mu_z

    res = res.T.reshape(src.shape)

    return res

def preprocess(src, ref, size, ncc):
    input_video = [np.array(Image.fromarray(c)) for c in src][:480]
    if not ncc:
        output_frames = []
        input_video_cc = [np.array(Image.fromarray(c).resize((256, 256))) for c in input_video]
        variables = vars(np.array(input_video_cc), ref)
        for i, frame in enumerate(input_video):
            img_res = transfer(frame, ref, variables)
            img_res = np.clip(img_res, 0, 255).astype(np.uint8)
            output_frames.append(img_res)
        input_video = output_frames

    input_video_resize = [np.array(Image.fromarray(c).resize((size, size), Image.Resampling.LANCZOS)) for c in input_video]

    return input_video, input_video_resize

def save_cube_lut(lut_array, path, size=16, title="AI_Generated_LUT"):
    lut_array = np.array(lut_array).flatten()
    with open(path, 'w') as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f'LUT_3D_SIZE {size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
        for i in range(0, len(lut_array), 3):
            f.write(f"{lut_array[i]:.6f} {lut_array[i+1]:.6f} {lut_array[i+2]:.6f}\n")

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

def upsample_lut(lut_flat, old_size=16, new_size=64):
    """
    Upsamples a 1D flattened 3D LUT using cubic spline interpolation 
    to dramatically improve color gradient rendering quality and prevent banding.
    Also applies a Gaussian filter to destroy AI-generated high-frequency grain.
    """
    # Pillow's Color3DLUT expects order to be B changing slowest, then G, then R
    lut_3d = lut_flat.reshape(old_size, old_size, old_size, 3)
    
    # Apply a low-pass Gaussian filter to denoise the AI output
    # This prevents grain and banding, guaranteeing professional-level smoothness
    for c in range(3):
        lut_3d[..., c] = gaussian_filter(lut_3d[..., c], sigma=0.4)
        
    grid_old = np.linspace(0, 1, old_size)
    
    interpolator = RegularGridInterpolator((grid_old, grid_old, grid_old), lut_3d, method='cubic')
    
    grid_new = np.linspace(0, 1, new_size)
    B, G, R = np.meshgrid(grid_new, grid_new, grid_new, indexing='ij')
    points = np.stack([B, G, R], axis=-1)
    
    upsampled_lut_3d = interpolator(points)
    upsampled_lut_flat = np.clip(upsampled_lut_3d, 0.0, 1.0).flatten()
    
    return upsampled_lut_flat
