#!/bin/bash
set -e

echo "Creating Python Virtual Environment with Python 3.9..."
rm -rf venv
/usr/bin/python3 -m venv venv
source venv/bin/activate

echo "Installing PyTorch for Mac..."
pip install --upgrade pip
pip install torch torchvision torchaudio

echo "Installing dependencies..."
pip install diffusers==0.21.4 transformers==4.32.0 tqdm omegaconf einops opencv-python pillow safetensors accelerate av imageio imageio-ffmpeg natsort pillow_lut huggingface_hub lpips piq image-quality scipy

echo "Downloading Pre-trained Models..."
mkdir -p pretrained_models
cd pretrained_models

if [ ! -d "stable-diffusion-v1-5" ]; then
    echo "Downloading Stable Diffusion v1.5..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
    cd stable-diffusion-v1-5
    git lfs pull
    cd ..
fi

if [ ! -d "clip-vit-base-patch32" ]; then
    echo "Downloading CLIP ViT-B/32..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/clip-vit-base-patch32
    cd clip-vit-base-patch32
    git lfs pull
    cd ..
fi

if [ ! -f "GS-Extractor.ckpt" ]; then
    echo "Downloading GS-Extractor..."
    curl -L -o GS-Extractor.ckpt https://huggingface.co/Kijai/VCG_comfy/resolve/main/checkpoints/GS-Extractor.ckpt
fi

if [ ! -f "L-Diffuser.ckpt" ]; then
    echo "Downloading L-Diffuser..."
    curl -L -o L-Diffuser.ckpt https://huggingface.co/Kijai/VCG_comfy/resolve/main/checkpoints/L-Diffuser.ckpt
fi

echo "Setup Complete!"
