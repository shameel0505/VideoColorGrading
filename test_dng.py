import os
import cv2
import rawpy
from PIL import Image
import numpy as np
from types import SimpleNamespace
from grading import Inference

def load_raw_image(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
    return rgb

# Setup
ref_path = "/Users/shameel/Downloads/pan-s-labyrinth_frame_14_thumb.webp"
target_path = "/Users/shameel/Downloads/PhotoTraces_Free_RAW_Photos_01_Manhattan_Skyline.dng"

print("Loading AI Models... This might take a minute...")
config_args = SimpleNamespace(config='configs/prompts/video_demo.yaml')
grader = Inference(config=config_args.config)
print("Models loaded successfully!")

print("Processing images...")
ref_pil = Image.open(ref_path).convert('RGB').resize((512, 512))
ref_sequence = np.array(ref_pil)

target_rgb = load_raw_image(target_path)
print("Running color grading inference...")
result_frames = grader(
    ref_sequence=ref_sequence, 
    input_frames=[target_rgb],
    return_frames=True,
    random_seed=42, 
    step=25, 
    size=512, 
    ncc=False
)

output_jpg = "final_output.jpg"
Image.fromarray(result_frames[0]).save(output_jpg, quality=100)
print(f"Success! Saved color graded image to {output_jpg}")
