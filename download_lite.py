from huggingface_hub import snapshot_download
import os
import shutil

print("Downloading SD 1.5 Lite...")
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="pretrained_models/stable-diffusion-v1-5",
    allow_patterns=[
        "*.json", 
        "*.txt", 
        "unet/diffusion_pytorch_model.bin", 
        "vae/diffusion_pytorch_model.bin", 
        "tokenizer/*", 
        "scheduler/*",
        "feature_extractor/*"
    ]
)

print("Downloading CLIP ViT-B/32...")
snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir="pretrained_models/clip-vit-base-patch32",
    allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "preprocessor_config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]
)

print("Downloading Kijai checkpoints...")
snapshot_download(
    repo_id="Kijai/VCG_comfy",
    local_dir="pretrained_models",
    allow_patterns=["checkpoints/GS-Extractor.ckpt", "checkpoints/L-Diffuser.ckpt"]
)

# Move Kijai checkpoints to the right location
os.rename("pretrained_models/checkpoints/GS-Extractor.ckpt", "pretrained_models/GS-Extractor.ckpt")
os.rename("pretrained_models/checkpoints/L-Diffuser.ckpt", "pretrained_models/L-Diffuser.ckpt")
shutil.rmtree("pretrained_models/checkpoints")

print("Downloads complete!")
