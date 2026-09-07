import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Enable CUDA memory allocation optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
from PIL import Image
import argparse
from types import SimpleNamespace
import tempfile
import uuid
import shutil
import subprocess
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import imageio_ffmpeg
from pillow_lut import load_cube_file

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Automatically select the best available device
if TORCH_AVAILABLE and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[GPU] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
elif TORCH_AVAILABLE:
    DEVICE = torch.device("cpu")
    print("[INFO] CUDA GPU not available -- using CPU")
else:
    DEVICE = "cpu"
    print("[INFO] PyTorch not installed in local environment -- running in Studio Color Science Mode")

if TORCH_AVAILABLE:
    torch.set_grad_enabled(False)

def check_nvenc():
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False

NVENC_AVAILABLE = check_nvenc()
print("[NVENC] NVENC hardware encoder available" if NVENC_AVAILABLE else "[NVENC] NVENC unavailable -- video encoding will use CPU")

app = FastAPI(title="CineGrade AI Pro Studio API")

# Allow CORS for local frontend dev & cloud tunnels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI models if available (Lazy Loaded)
grader = None
model_load_error = None
is_model_loaded = False

def lazy_load_grader():
    global grader, is_model_loaded, model_load_error
    if is_model_loaded: return grader
    
    try:
        print("Checking AI Pipeline Diffusion Models...")
        from grading import Inference
        config_args = SimpleNamespace(config='configs/prompts/video_demo.yaml')
        grader = Inference(config=config_args.config)
        
        # Move PyTorch model components to GPU when possible
        if hasattr(DEVICE, "type") and DEVICE.type == "cuda":
            try:
                if hasattr(grader, "model"):
                    grader.model = grader.model.to(DEVICE)
                if hasattr(grader, "pipeline"):
                    grader.pipeline = grader.pipeline.to(DEVICE)
                print(f"[AI] AI grader moved to {DEVICE}")
            except Exception as e:
                print(f"[WARNING] Could not explicitly move grader to GPU: {e}")

        print("AI Diffusion Models loaded successfully!")
    except Exception as e:
        print(f"Notice: Running in studio 3D LUT Color Transfer & Auto-Grader mode ({e})")
        grader = None
        model_load_error = e
    finally:
        is_model_loaded = True
        
    return grader

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "CineGrade AI Studio API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "CineGrade AI API"}

@app.get("/api/library")
async def get_library():
    ref_dir = "cinematic_references"
    if not os.path.exists(ref_dir):
        return []
    
    images = []
    # Curated category mapping for professional movie looks
    categories = {
        "Dune": "Sci-Fi / Desert",
        "Blade_Runner": "Cyberpunk / Neon",
        "Oppenheimer": "Cinematic / Drama",
        "The_Matrix": "Sci-Fi / Green Noir",
        "The_Batman": "Dark / Noir",
        "Interstellar": "Deep Space / Warm",
        "La_La_Land": "Vibrant / Pastel",
        "The_Grand_Budapest_Hotel": "Stylized / Pastel",
        "Mad_Max": "High Contrast / Warm",
        "Moonlight": "Neon / Indigo",
        "In_the_Mood_for_Love": "Vintage / Warm Red",
        "Her": "Warm / Pastel Red",
        "Amelie": "Warm / Golden Green",
        "Drive": "Synthwave / Neon",
        "Arrival": "Muted / Cold Sci-Fi",
        "John_Wick": "Action / High Contrast Cyan",
        "Se7en": "Gritty / Desaturated",
        "Prisoners": "Nordic / Cold Gloom",
        "The_Revenant": "Natural Light / Cold",
        "Fargo": "Winter / Crisp",
        "No_Country_for_Old_Men": "Bleach Bypass / Desert"
    }

    for f in sorted(os.listdir(ref_dir)):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            clean_name = f.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
            matched_cat = "Cinematic Look"
            for prefix, cat in categories.items():
                if f.startswith(prefix):
                    matched_cat = cat
                    break
            images.append({
                "id": clean_name,
                "name": clean_name.replace('_', ' '),
                "category": matched_cat,
                "path": f"/api/library/{f}"
            })
    return images

@app.get("/api/library/{filename}")
async def serve_library_image(filename: str):
    file_path = os.path.join("cinematic_references", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")

tasks = {}

# ==============================================================================
# 🎨 Studio-Grade Oklab & Subtractive Film Color Science Engine
# ==============================================================================

# Forward & Inverse Matrices for Linear sRGB <-> Oklab (Björn Ottosson, 2020)
_OKLAB_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
], dtype=np.float32)

_OKLAB_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
], dtype=np.float32)

_OKLAB_M2_INV = np.linalg.inv(_OKLAB_M2).astype(np.float32)
_OKLAB_M1_INV = np.linalg.inv(_OKLAB_M1).astype(np.float32)

def srgb_to_linear(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    mask = rgb <= 0.04045
    linear = np.empty_like(rgb)
    linear[mask] = rgb[mask] / 12.92
    linear[~mask] = np.power((rgb[~mask] + 0.055) / 1.055, 2.4)
    return linear

def linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    mask = linear <= 0.0031308
    srgb = np.empty_like(linear)
    srgb[mask] = 12.92 * linear[mask]
    srgb[~mask] = 1.055 * np.power(np.maximum(linear[~mask], 0.0), 1.0 / 2.4) - 0.055
    return np.clip(srgb, 0.0, 1.0)

def rgb_to_oklab(rgb):
    lin = srgb_to_linear(rgb)
    shape = lin.shape
    lin_flat = lin.reshape(-1, 3)
    lms = lin_flat @ _OKLAB_M1.T
    lms = np.maximum(lms, 0.0)
    lms_prime = np.cbrt(lms)
    oklab = lms_prime @ _OKLAB_M2.T
    return oklab.reshape(shape)

def oklab_to_rgb(oklab):
    shape = oklab.shape
    oklab_flat = oklab.reshape(-1, 3)
    lms_prime = oklab_flat @ _OKLAB_M2_INV.T
    lms = np.maximum(lms_prime, 0.0) ** 3
    lin = lms @ _OKLAB_M1_INV.T
    lin = np.clip(lin, 0.0, 1.0)
    srgb = linear_to_srgb(lin.reshape(shape))
    return srgb

def compute_oklab_skin_mask(oklab):
    """Calculates continuous Melanin Skin Tone probability in Oklab space."""
    L = oklab[:, 0]
    a = oklab[:, 1]
    b = oklab[:, 2]
    
    chroma = np.sqrt(a**2 + b**2)
    hue = np.arctan2(b, a) # Hue angle in radians
    
    # Human melanin skin locus in Oklab: ~0.95 rad (~54 degrees)
    hue_center = 0.95
    hue_width = 0.35
    hue_weight = np.clip(1.0 - (np.abs(hue - hue_center) / hue_width)**2, 0.0, 1.0)
    
    chroma_weight = np.clip((chroma - 0.03) / 0.05, 0.0, 1.0) * np.clip((0.20 - chroma) / 0.06, 0.0, 1.0)
    lightness_weight = np.clip((L - 0.20) / 0.15, 0.0, 1.0) * np.clip((0.88 - L) / 0.15, 0.0, 1.0)
    
    return np.clip(hue_weight * chroma_weight * lightness_weight, 0.0, 1.0)

def compute_skin_mask_vectorized(rgb_flt):
    oklab = rgb_to_oklab(rgb_flt)
    return compute_oklab_skin_mask(oklab)

def compute_skin_mask(rgb_np):
    h, w, c = rgb_np.shape
    flt = (rgb_np.reshape(-1, 3).astype(np.float32) / 255.0)
    return compute_skin_mask_vectorized(flt).reshape(h, w)

def generate_pro_reference_lut(ref_np, target_np, output_cube_path, lut_size=33, intensity=1.0, protect_skin=True):
    """
    Studio-Grade Cinematic Reference Matcher in Oklab:
    - Monotonic Luminance Transfer with Boundary Anchors (no crushing, no clipping)
    - Subtractive Color Density (CMY film light absorption)
    - Zone-aware smooth chromatic translation (Shadows, Midtones, Highlights)
    - Melanin Skin Tone Lock
    """
    ref_rgb = ref_np.astype(np.float32) / 255.0
    tgt_rgb = target_np.astype(np.float32) / 255.0
    
    ref_oklab = rgb_to_oklab(ref_rgb)
    tgt_oklab = rgb_to_oklab(tgt_rgb)
    
    ref_l = ref_oklab[:, :, 0].flatten()
    tgt_l = tgt_oklab[:, :, 0].flatten()
    
    # 1. Monotonic Luminance Quantile Matching
    quantiles = np.linspace(0, 100, 101)
    tgt_l_q = np.percentile(tgt_l, quantiles)
    ref_l_q = np.percentile(ref_l, quantiles)
    
    tgt_l_q = np.concatenate([[0.0], tgt_l_q, [1.0]])
    ref_l_q = np.concatenate([[0.0], ref_l_q, [1.0]])
    
    tgt_l_q_u, u_idx = np.unique(tgt_l_q, return_index=True)
    ref_l_q_u = ref_l_q[u_idx]
    
    # 2. Extract Zone-based Reference Chromatic Palette in Oklab
    ref_l_2d = ref_oklab[:, :, 0]
    sh_mask = ref_l_2d < 0.35
    hi_mask = ref_l_2d > 0.65
    mid_mask = (~sh_mask) & (~hi_mask)
    
    ref_ab_sh = np.median(ref_oklab[sh_mask, 1:3], axis=0) if np.sum(sh_mask) > 50 else np.array([0.0, 0.0], dtype=np.float32)
    ref_ab_mid = np.median(ref_oklab[mid_mask, 1:3], axis=0) if np.sum(mid_mask) > 50 else np.array([0.0, 0.0], dtype=np.float32)
    ref_ab_hi = np.median(ref_oklab[hi_mask, 1:3], axis=0) if np.sum(hi_mask) > 50 else np.array([0.0, 0.0], dtype=np.float32)
    
    # Chromatic dispersion
    ref_chroma_std = np.std(np.sqrt(ref_oklab[:, :, 1]**2 + ref_oklab[:, :, 2]**2)) + 1e-5
    tgt_chroma_std = np.std(np.sqrt(tgt_oklab[:, :, 1]**2 + tgt_oklab[:, :, 2]**2)) + 1e-5
    chroma_scale = np.clip(ref_chroma_std / tgt_chroma_std, 0.7, 1.4)
    
    # 3. Build 3D LUT lattice
    b_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    g_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    r_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    
    B, G, R = np.meshgrid(b_vals, g_vals, r_vals, indexing='ij')
    rgb_lattice = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    
    oklab_lattice = rgb_to_oklab(rgb_lattice)
    orig_l = oklab_lattice[:, 0]
    
    matched_l = np.interp(orig_l, tgt_l_q_u, ref_l_q_u)
    graded_l = orig_l * 0.35 + matched_l * 0.65
    
    # Zone weights for smooth transitions
    w_sh = np.clip((0.38 - graded_l) / 0.35, 0.0, 1.0) ** 1.6
    w_hi = np.clip((graded_l - 0.55) / 0.35, 0.0, 1.0) ** 1.6
    w_mid = np.maximum(0.0, 1.0 - w_sh - w_hi)
    tot = w_sh + w_mid + w_hi + 1e-6
    w_sh /= tot
    w_mid /= tot
    w_hi /= tot
    
    target_ab = w_sh[:, None] * ref_ab_sh + w_mid[:, None] * ref_ab_mid + w_hi[:, None] * ref_ab_hi
    
    # Chromatic blend in Oklab
    graded_a = oklab_lattice[:, 1] * chroma_scale * 0.60 + target_ab[:, 0] * 0.40
    graded_b = oklab_lattice[:, 2] * chroma_scale * 0.60 + target_ab[:, 1] * 0.40
    
    # Subtractive Color Density (more saturated = deeper light absorption)
    C = np.sqrt(graded_a**2 + graded_b**2)
    graded_l = graded_l * (1.0 - 0.25 * (C ** 1.2))
    
    # Parabolic Highlight Roll-Off (prevents any digital clipping)
    graded_l = np.where(graded_l > 0.88, 0.88 + 0.12 * (1.0 - np.exp(-(graded_l - 0.88) / 0.12)), graded_l)
    graded_l = np.clip(graded_l, 0.0, 1.0)
    
    graded_oklab = np.stack([graded_l, graded_a, graded_b], axis=-1)
    
    # Melanin Skin Protection
    if protect_skin:
        skin_probs = compute_oklab_skin_mask(oklab_lattice)
        blend_mask = (skin_probs * 0.80)[:, None]
        skin_natural = np.stack([orig_l * 0.50 + graded_l * 0.50, oklab_lattice[:, 1] * 1.02, oklab_lattice[:, 2] * 1.04], axis=-1)
        graded_oklab = graded_oklab * (1.0 - blend_mask) + skin_natural * blend_mask
        
    graded_rgb = oklab_to_rgb(graded_oklab)
    
    # Anchored clean pure black & white
    graded_rgb[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    graded_rgb[-1] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    final_rgb = rgb_lattice * (1.0 - intensity) + graded_rgb * intensity
    final_rgb = np.clip(final_rgb, 0.0, 1.0)
    
    with open(output_cube_path, 'w') as f:
        f.write('TITLE "CineGrade Pro Oklab Reference Matcher"\n')
        f.write(f'LUT_3D_SIZE {lut_size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
        for i in range(len(final_rgb)):
            f.write(f'{final_rgb[i, 0]:.6f} {final_rgb[i, 1]:.6f} {final_rgb[i, 2]:.6f}\n')

def generate_auto_grade_lut(target_np, output_cube_path, lut_size=33, intensity=1.0, style="vision3", is_log=False):
    """
    Photographic Master Film Stock Emulation Engine (Oklab + Subtractive CMY Density):
    - 'vision3' / 'blockbuster': Kodak Vision3 500T (Hollywood Teal & Gold)
    - 'portra' / 'golden_hour': Kodak Portra 400 (Warm & Luminous Skin)
    - 'eterna' / 'noir': Fuji Eterna 8543 (Muted Film Noir)
    - 'commercial' / 'clean': Clean Commercial 35mm (Crisp & Vibrant)
    """
    b_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    g_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    r_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    
    B, G, R = np.meshgrid(b_vals, g_vals, r_vals, indexing='ij')
    rgb_lattice = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    
    rgb_in = rgb_lattice.copy()
    if is_log:
        rgb_in = np.power(np.maximum(rgb_in - 0.05, 0.0) / 0.90, 1.45)
        rgb_in = np.clip(rgb_in, 0.0, 1.0)
        
    oklab = rgb_to_oklab(rgb_in)
    L = oklab[:, 0]
    a = oklab[:, 1]
    b = oklab[:, 2]
    
    # Authentic Film Toe & S-Curve
    L_norm = np.clip(L, 0.0, 1.0)
    L_curved = (L_norm * (2.35 * L_norm + 0.04)) / (L_norm * (2.25 * L_norm + 0.65) + 0.10)
    L_curved = np.clip(L_curved, 0.0, 1.0)
    
    C = np.sqrt(a**2 + b**2)
    
    if style in ["vision3", "blockbuster"]:
        # Kodak Vision3 500T 5219: Deep oceanic teal in shadows, warm golden highlights
        L_target = L_curved * 0.75 + L_norm * 0.25
        L_target = L_target * (1.0 - 0.30 * (C ** 1.2)) # Subtractive density
        
        sh_w = np.clip((0.40 - L_target) / 0.40, 0.0, 1.0) ** 1.6
        hi_w = np.clip((L_target - 0.50) / 0.50, 0.0, 1.0) ** 1.6
        
        a_graded = a - 0.018 * sh_w + 0.016 * hi_w
        b_graded = b - 0.030 * sh_w + 0.032 * hi_w
        
    elif style in ["portra", "golden_hour"]:
        # Kodak Portra 400: Luminous warm midtones, gentle shadow lift, flattering complexion
        L_target = L_curved * 0.60 + L_norm * 0.40
        L_target = L_target + 0.02 * np.clip((0.30 - L_target) / 0.30, 0.0, 1.0) # Shadow lift
        L_target = L_target * (1.0 - 0.22 * (C ** 1.2))
        
        hi_w = np.clip((L_target - 0.45) / 0.55, 0.0, 1.0) ** 1.5
        sh_w = np.clip((0.35 - L_target) / 0.35, 0.0, 1.0) ** 1.5
        
        a_graded = a + 0.012 * hi_w + 0.005 * sh_w
        b_graded = b + 0.028 * hi_w - 0.010 * sh_w
        
    elif style in ["eterna", "noir"]:
        # Fuji Eterna 8543: Cool slate shadows, restrained saturation, moody Nordic contrast
        L_target = L_curved * 0.85 + L_norm * 0.15
        L_target = L_target * (1.0 - 0.25 * (C ** 1.2))
        
        sh_w = np.clip((0.45 - L_target) / 0.45, 0.0, 1.0) ** 1.5
        hi_w = np.clip((L_target - 0.55) / 0.45, 0.0, 1.0) ** 1.5
        
        a_graded = (a - 0.012 * sh_w) * 0.88
        b_graded = (b - 0.015 * sh_w) * 0.88
        
    elif style == "cinestill":
        # CineStill 800T: Iconic tungsten night look, deep cobalt shadows, glowing warm tungsten highlights
        L_target = L_curved * 0.80 + L_norm * 0.20
        L_target = L_target * (1.0 - 0.32 * (C ** 1.2))
        
        sh_w = np.clip((0.38 - L_target) / 0.38, 0.0, 1.0) ** 1.5
        hi_w = np.clip((L_target - 0.48) / 0.52, 0.0, 1.0) ** 1.5
        
        a_graded = a - 0.012 * sh_w + 0.022 * hi_w
        b_graded = b - 0.038 * sh_w + 0.040 * hi_w

    elif style == "kodachrome":
        # Kodak Kodachrome 64: Vintage 1970s National Geographic, rich punchy contrast, saturated primary reds & greens
        L_target = L_curved * 0.90 + L_norm * 0.10
        L_target = L_target * (1.0 - 0.26 * (C ** 1.2))
        
        hi_w = np.clip((L_target - 0.55) / 0.45, 0.0, 1.0) ** 1.4
        sh_w = np.clip((0.35 - L_target) / 0.35, 0.0, 1.0) ** 1.4
        
        a_graded = a * 1.18 + 0.008 * hi_w
        b_graded = b * 1.14 - 0.006 * sh_w

    elif style == "fuji_pro":
        # Fujifilm Pro 400H: Soft pastel editorial, airy mint greens, cyan skies, luminous creamy highlights
        L_target = L_curved * 0.55 + L_norm * 0.45 + 0.025 # Airy midtone lift
        L_target = L_target * (1.0 - 0.20 * (C ** 1.2))
        
        hi_w = np.clip((L_target - 0.50) / 0.50, 0.0, 1.0) ** 1.5
        sh_w = np.clip((0.40 - L_target) / 0.40, 0.0, 1.0) ** 1.5
        
        a_graded = a * 0.92 - 0.012 * hi_w - 0.006 * sh_w
        b_graded = b * 0.94 - 0.015 * hi_w + 0.004 * sh_w

    elif style == "bleach_bypass":
        # Bleach Bypass (Silver Retention): High local contrast, desaturated silver tones, gritty action thriller
        L_target = np.clip(L_curved * 1.05 - 0.02, 0.0, 1.0)
        L_target = L_target * (1.0 - 0.15 * (C ** 1.2))
        
        sh_w = np.clip((0.40 - L_target) / 0.40, 0.0, 1.0) ** 1.5
        
        a_graded = (a - 0.006 * sh_w) * 0.50 # Heavy desaturation
        b_graded = (b - 0.008 * sh_w) * 0.50

    else: # "commercial" / "clean"
        # Clean Commercial 35mm: Vivid memory colors, punchy broadcast dynamic range
        L_target = L_curved * 0.50 + L_norm * 0.50
        L_target = L_target * (1.0 - 0.28 * (C ** 1.2))
        
        hi_w = np.clip((L_target - 0.60) / 0.40, 0.0, 1.0) ** 1.5
        sh_w = np.clip((0.35 - L_target) / 0.35, 0.0, 1.0) ** 1.5
        
        a_graded = a * 1.10 + 0.006 * hi_w
        b_graded = b * 1.10 - 0.008 * sh_w
        
    # Parabolic Highlight Roll-Off
    L_target = np.where(L_target > 0.85, 0.85 + 0.15 * (1.0 - np.exp(-(L_target - 0.85) / 0.15)), L_target)
    L_target = np.clip(L_target, 0.0, 1.0)
    
    # Melanin Skin Tone Lock
    skin_probs = compute_oklab_skin_mask(oklab)
    skin_blend = (skin_probs * 0.80)[:, None]
    
    graded_oklab = np.stack([L_target, a_graded, b_graded], axis=-1)
    
    skin_natural_L = L_norm * 0.70 + L_curved * 0.30
    skin_natural_oklab = np.stack([skin_natural_L, a * 1.02, b * 1.04], axis=-1)
    
    final_oklab = graded_oklab * (1.0 - skin_blend) + skin_natural_oklab * skin_blend
    graded_rgb = oklab_to_rgb(final_oklab)
    
    # Anchored pure black & white
    graded_rgb[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    graded_rgb[-1] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    final_rgb = rgb_lattice * (1.0 - intensity) + graded_rgb * intensity
    final_rgb = np.clip(final_rgb, 0.0, 1.0)
    
    with open(output_cube_path, 'w') as f:
        f.write(f'TITLE "CineGrade Master Film Stock - {style.upper()}"\n')
        f.write(f'LUT_3D_SIZE {lut_size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
        for i in range(len(final_rgb)):
            f.write(f'{final_rgb[i, 0]:.6f} {final_rgb[i, 1]:.6f} {final_rgb[i, 2]:.6f}\n')

def extract_embedded_jpeg_from_raw(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = f.read(25 * 1024 * 1024)
            soi_idx = data.find(b'\xff\xd8\xff')
            if soi_idx != -1:
                eoi_idx = data.find(b'\xff\xd9', soi_idx)
                if eoi_idx != -1 and (eoi_idx - soi_idx) > 5000:
                    import io
                    jpeg_bytes = data[soi_idx:eoi_idx+2]
                    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
                    return np.array(img)
    except Exception:
        pass
    return None

def run_grading_task(uid, ref_path, target_path, is_video, steps, size, ncc, output_cube, intensity=1.0, protect_skin=True, mode="reference", style="vision3", is_log=False):
    tasks[uid] = {"status": "processing"}
    try:
        def load_image_with_raw_support(filepath):
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.dng', '.cr2', '.nef', '.arw', '.ari']:
                try:
                    import rawpy
                    with rawpy.imread(filepath) as raw:
                        return raw.postprocess(use_camera_wb=True)
                except Exception:
                    pass
            if ext in ['.braw', '.r3d', '.ari']:
                emb = extract_embedded_jpeg_from_raw(filepath)
                if emb is not None:
                    return emb
            try:
                return np.array(Image.open(filepath).convert('RGB'))
            except Exception:
                emb = extract_embedded_jpeg_from_raw(filepath)
                if emb is not None:
                    return emb
                raise

        if not is_video:
            target_image = load_image_with_raw_support(target_path)
            target_thumb = np.array(Image.fromarray(target_image).resize((512, 512), Image.Resampling.LANCZOS))
            
            if mode == "auto":
                generate_auto_grade_lut(target_thumb, output_cube, intensity=intensity, style=style, is_log=is_log)
            else:
                reference_image = Image.open(ref_path).convert('RGB').resize((size, size))
                reference_image_np = np.array(reference_image)
                
                active_grader = lazy_load_grader()
                if active_grader is not None and intensity >= 0.95 and not protect_skin:
                    active_grader(
                        ref_sequence=reference_image_np,
                        input_frames=[target_thumb],
                        return_frames=False,
                        save_lut_path=output_cube,
                        random_seed=42, 
                        step=steps, 
                        size=size, 
                        ncc=ncc
                    )
                else:
                    generate_pro_reference_lut(
                        reference_image_np, 
                        target_thumb, 
                        output_cube, 
                        intensity=intensity, 
                        protect_skin=protect_skin
                    )
            
            # Save original JPG for before/after comparison
            orig_jpg = os.path.join(tempfile.gettempdir(), f"original_{uid}.jpg")
            Image.fromarray(target_image).save(orig_jpg, quality=95)
            
            # Apply LUT to image
            lut = load_cube_file(output_cube)
            output_jpg = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.jpg")
            Image.fromarray(target_image).filter(lut).save(output_jpg, quality=100)
            
            tasks[uid] = {
                "status": "completed", 
                "result": {
                    "output_media": output_jpg,
                    "original_media": orig_jpg,
                    "output_lut": output_cube, 
                    "type": "image"
                }
            }
        else:
            output_mp4 = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.mp4")
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Robust multi-strategy frame extraction (handles short clips, 4K/8K, HEVC, ProRes, MKV, VFR)
            frame_rgb = None
            frame_jpg = os.path.join(tempfile.gettempdir(), f"frame_{uid}.jpg")
            
            # Strategy 1: Direct first-frame extraction
            try:
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i", target_path,
                    "-vframes", "1",
                    "-q:v", "2",
                    frame_jpg
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                if os.path.exists(frame_jpg) and os.path.getsize(frame_jpg) > 0:
                    frame_rgb = np.array(Image.open(frame_jpg).convert('RGB'))
                    try:
                        os.remove(frame_jpg)
                    except Exception:
                        pass
            except Exception:
                frame_rgb = None
                
            # Strategy 2: Fast seek extraction (for long videos)
            if frame_rgb is None:
                try:
                    cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-ss", "0.5",
                        "-i", target_path,
                        "-vframes", "1",
                        "-q:v", "2",
                        frame_jpg
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                    if os.path.exists(frame_jpg) and os.path.getsize(frame_jpg) > 0:
                        frame_rgb = np.array(Image.open(frame_jpg).convert('RGB'))
                        try:
                            os.remove(frame_jpg)
                        except Exception:
                            pass
                except Exception:
                    frame_rgb = None

            # Strategy 3: PyAV / imageio reader
            if frame_rgb is None:
                try:
                    import imageio.v3 as iio
                    frame_rgb = iio.imread(target_path, index=0)
                except Exception:
                    frame_rgb = None

            # Strategy 4: OpenCV VideoCapture fallback
            if frame_rgb is None:
                try:
                    cap = cv2.VideoCapture(target_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    frame_rgb = None
            
            if frame_rgb is None:
                ext = target_ext.lower()
                if ext in ['.braw', '.r3d', '.ari']:
                    raise Exception(f"{ext.upper()} is a proprietary cinema RAW video format that requires sensor debayering. Please export a still (PNG, JPG, TIFF) or ProRes/MP4 clip from DaVinci Resolve or Premiere, drop it into CineGrade to generate your custom .CUBE 3D LUT, and apply the LUT to your raw timeline.")
                else:
                    raise Exception("Could not extract a representative frame from the video footage.")
            
            if mode == "auto":
                generate_auto_grade_lut(frame_rgb, output_cube, intensity=intensity, style=style, is_log=is_log)
            else:
                reference_image = Image.open(ref_path).convert('RGB').resize((size, size))
                reference_image_np = np.array(reference_image)
                
                active_grader = lazy_load_grader()
                if active_grader is not None and intensity >= 0.95 and not protect_skin:
                    active_grader(
                        ref_sequence=reference_image_np,
                        input_frames=[frame_rgb],
                        return_frames=False,
                        save_lut_path=output_cube,
                        random_seed=42, 
                        step=steps, 
                        size=size, 
                        ncc=ncc
                    )
                else:
                    generate_pro_reference_lut(
                        reference_image_np, 
                        frame_rgb, 
                        output_cube, 
                        intensity=intensity, 
                        protect_skin=protect_skin
                    )
            
            cube_dir = os.path.dirname(output_cube)
            cube_name = os.path.basename(output_cube)
            
            # Robust multi-pass encoding: GPU NVENC with auto-fallback to high-speed CPU libx264,
            # and audio copy with auto-fallback to AAC (prevents PCM audio crashes on MP4 container).
            encoders_to_try = []
            if TORCH_AVAILABLE and torch.cuda.is_available() and NVENC_AVAILABLE:
                encoders_to_try.append({
                    "name": "h264_nvenc",
                    "args": ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-gpu", "0"]
                })
            encoders_to_try.append({
                "name": "libx264",
                "args": ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "0"]
            })

            render_success = False
            last_ffmpeg_err = ""

            for enc in encoders_to_try:
                if render_success:
                    break
                for audio_args in [["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]]:
                    ffmpeg_cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-i", target_path,
                        "-vf", f"lut3d={cube_name}",
                        *audio_args,
                        *enc["args"],
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        output_mp4
                    ]
                    
                    try:
                        # Avoid pipe deadlock by discarding stdout and capturing stderr with timeout
                        proc = subprocess.run(
                            ffmpeg_cmd,
                            cwd=cube_dir,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            timeout=7200
                        )
                        if proc.returncode == 0 and os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 0:
                            render_success = True
                            break
                        else:
                            last_ffmpeg_err = proc.stderr.decode('utf-8', errors='ignore')
                    except Exception as enc_err:
                        last_ffmpeg_err = str(enc_err)

            if not render_success:
                if frame_rgb is not None:
                    # Graceful cinema raw fallback: generate graded preview still and export .CUBE 3D LUT
                    lut = load_cube_file(output_cube)
                    orig_jpg = os.path.join(tempfile.gettempdir(), f"original_{uid}.jpg")
                    Image.fromarray(frame_rgb).save(orig_jpg, quality=95)
                    output_jpg = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.jpg")
                    Image.fromarray(frame_rgb).filter(lut).save(output_jpg, quality=100)
                    tasks[uid] = {
                        "status": "completed",
                        "result": {
                            "output_media": output_jpg,
                            "original_media": orig_jpg,
                            "output_lut": output_cube,
                            "type": "image"
                        }
                    }
                    return
                else:
                    tasks[uid] = {"status": "error", "error": f"Video encoding failed: {last_ffmpeg_err[-400:]}"}
                    return
                
            tasks[uid] = {
                "status": "completed", 
                "result": {
                    "output_media": output_mp4,
                    "original_media": target_path,
                    "output_lut": output_cube, 
                    "type": "video"
                }
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        tasks[uid] = {"status": "error", "error": str(e)}

@app.get("/api/gpu")
def gpu_status():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda_available": True,
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(props.total_memory / 1024**3, 2),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            "nvenc_available": NVENC_AVAILABLE
        }
    return {
        "cuda_available": False,
        "device": "cpu",
        "nvenc_available": False
    }

@app.post("/api/grade")
def process_grading(
    background_tasks: BackgroundTasks,
    target: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    ref_id: Optional[str] = Form(None),
    mode: str = Form("reference"), # "reference" or "auto"
    style: str = Form("vision3"), # "vision3", "portra", "eterna", "commercial"
    intensity: float = Form(1.0),
    protect_skin: bool = Form(True),
    is_log: bool = Form(False),
    steps: int = Form(25),
    size: int = Form(512),
    ncc: bool = Form(True)
):
    uid = uuid.uuid4().hex[:8]
    
    ref_path = None
    if mode == "reference":
        if reference is not None and getattr(reference, 'filename', None):
            ref_ext = os.path.splitext(reference.filename)[1].lower()
            ref_path = os.path.join(tempfile.gettempdir(), f"ref_{uid}{ref_ext}")
            with open(ref_path, "wb") as f:
                shutil.copyfileobj(reference.file, f)
        elif ref_id:
            for ext in ['.jpg', '.jpeg', '.png']:
                possible_path = os.path.join("cinematic_references", f"{ref_id}{ext}")
                if os.path.exists(possible_path):
                    ref_path = possible_path
                    break
        
        if not ref_path:
            raise HTTPException(status_code=400, detail="Reference image or valid ref_id is required.")
        
    target_ext = os.path.splitext(target.filename)[1].lower()
    target_path = os.path.join(tempfile.gettempdir(), f"target_{uid}{target_ext}")
    with open(target_path, "wb") as f:
        shutil.copyfileobj(target.file, f)
        
    output_cube = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.cube")
    is_video = target_ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v', '.ts', '.mts', '.m2ts', '.r3d', '.braw', '.ari']
    
    background_tasks.add_task(
        run_grading_task, 
        uid, 
        ref_path, 
        target_path, 
        is_video, 
        steps, 
        size, 
        ncc, 
        output_cube, 
        intensity, 
        protect_skin,
        mode,
        style,
        is_log
    )
    
    return {"task_id": uid, "status": "processing"}

@app.get("/api/status/{task_id}")
def check_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.api_route("/api/download", methods=["GET", "HEAD"])
def download_by_query(path: str):
    target_path = path
    if not os.path.exists(target_path):
        target_path = os.path.join(tempfile.gettempdir(), os.path.basename(path))
        
    if os.path.exists(target_path):
        media_type = "video/mp4" if target_path.endswith(".mp4") else (
            "image/jpeg" if target_path.endswith((".jpg", ".jpeg")) else "application/octet-stream"
        )
        fname = os.path.basename(target_path)
        return FileResponse(
            target_path, 
            media_type=media_type, 
            filename=fname,
            headers={"Content-Disposition": f'attachment; filename="{fname}"'}
        )
    raise HTTPException(status_code=404, detail=f"File not found: {path}")

@app.api_route("/api/download/{filename}", methods=["GET", "HEAD"])
def download_file(filename: str):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(file_path):
        media_type = "video/mp4" if filename.endswith(".mp4") else (
            "image/jpeg" if filename.endswith((".jpg", ".jpeg")) else "application/octet-stream"
        )
        return FileResponse(
            file_path, 
            media_type=media_type, 
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8444, reload=False)
