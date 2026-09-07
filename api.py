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

def compute_skin_mask_vectorized(rgb_flt):
    """
    Vectorized high-precision skin locus probability computation.
    rgb_flt: (N, 3) float in [0, 1]
    returns: (N,) skin probability in [0, 1]
    """
    rgb_u8 = np.clip(rgb_flt * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    ycrcb = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).reshape(-1, 3).astype(np.float32)
    y = ycrcb[:, 0] / 255.0
    cr = ycrcb[:, 1]
    cb = ycrcb[:, 2]
    
    # Human skin tone locus on I-line: center ~ 152 Cr, 110 Cb
    d_cr = (cr - 152.0) / 18.0
    d_cb = (cb - 110.0) / 15.0
    dist_sq = d_cr**2 + d_cb**2
    skin_prob = np.exp(-0.5 * dist_sq)
    
    # Luminance gating to exclude deep blacks and blown whites
    lum_gate = np.clip((y - 0.12) / 0.12, 0.0, 1.0) * np.clip((0.92 - y) / 0.12, 0.0, 1.0)
    return np.clip(skin_prob * lum_gate, 0.0, 1.0)

def compute_skin_mask(rgb_np):
    """Generates a soft probability mask (0.0 to 1.0) of human skin tones for image arrays."""
    h, w, c = rgb_np.shape
    flt = (rgb_np.reshape(-1, 3).astype(np.float32) / 255.0)
    return compute_skin_mask_vectorized(flt).reshape(h, w)

def generate_pro_reference_lut(ref_np, target_np, output_cube_path, lut_size=33, intensity=1.0, protect_skin=True):
    """
    Studio-Grade Cinematic Reference Matcher:
    - Anchored dynamic range (clean blacks, protected highlights, smooth film midtones)
    - Monotonic smooth luminance transfer
    - Organic Split-Toning chromatic alignment based on reference shadow/mid/highlight palettes
    - Vectorized Skin Tone Line protection
    """
    ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    ref_l = ref_lab[:, :, 0].flatten()
    tgt_l = tgt_lab[:, :, 0].flatten()
    
    # 1. Luminance Quantile Transfer with Zero & Max Anchors
    quantiles = np.linspace(0, 100, 101)
    tgt_l_q = np.percentile(tgt_l, quantiles)
    ref_l_q = np.percentile(ref_l, quantiles)
    
    # Ensure strict monotonicity and boundary anchors
    tgt_l_q = np.concatenate([[0.0], tgt_l_q, [255.0]])
    ref_l_q = np.concatenate([[0.0], ref_l_q, [255.0]])
    
    tgt_l_q_unique, unique_indices = np.unique(tgt_l_q, return_index=True)
    ref_l_q_unique = ref_l_q[unique_indices]
    
    # 2. Extract Reference Chromatic Tones by Luminance Zone
    l_ref_2d = ref_lab[:, :, 0]
    shadow_mask = l_ref_2d < 80
    high_mask = l_ref_2d > 175
    mid_mask = (~shadow_mask) & (~high_mask)
    
    ref_ab_shadow = np.median(ref_lab[shadow_mask, 1:3], axis=0) if np.sum(shadow_mask) > 50 else np.array([128.0, 128.0], dtype=np.float32)
    ref_ab_mid = np.median(ref_lab[mid_mask, 1:3], axis=0) if np.sum(mid_mask) > 50 else np.array([128.0, 128.0], dtype=np.float32)
    ref_ab_high = np.median(ref_lab[high_mask, 1:3], axis=0) if np.sum(high_mask) > 50 else np.array([128.0, 128.0], dtype=np.float32)
    
    # Calculate gentle chromatic offsets from neutral 128
    d_shadow = (ref_ab_shadow - 128.0) * 0.45
    d_mid = (ref_ab_mid - 128.0) * 0.35
    d_high = (ref_ab_high - 128.0) * 0.40
    
    # 3. Build 3D LUT lattice
    b_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    g_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    r_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    
    B, G, R = np.meshgrid(b_vals, g_vals, r_vals, indexing='ij')
    rgb_lattice = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    
    rgb_lattice_u8 = (rgb_lattice * 255.0).astype(np.uint8).reshape(-1, 1, 3)
    lab_lattice = cv2.cvtColor(rgb_lattice_u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    
    # Smooth luminance transfer
    orig_l = lab_lattice[:, 0]
    matched_l = np.interp(orig_l, tgt_l_q_unique, ref_l_q_unique)
    graded_l = orig_l * 0.35 + matched_l * 0.65
    lab_lattice[:, 0] = np.clip(graded_l, 0.0, 255.0)
    
    # Smooth zone weights for chromatic split-toning
    norm_l = lab_lattice[:, 0] / 255.0
    w_shadow = np.clip((0.40 - norm_l) / 0.35, 0.0, 1.0) ** 1.8
    w_high = np.clip((norm_l - 0.60) / 0.35, 0.0, 1.0) ** 1.8
    w_mid = np.maximum(0.0, 1.0 - w_shadow - w_high)
    tot = w_shadow + w_mid + w_high + 1e-6
    w_shadow /= tot
    w_mid /= tot
    w_high /= tot
    
    chroma_shift = (
        w_shadow[:, None] * d_shadow +
        w_mid[:, None] * d_mid +
        w_high[:, None] * d_high
    )
    
    # Apply chromatic shift with smooth saturation gating
    current_ab = lab_lattice[:, 1:3]
    lab_lattice[:, 1:3] = np.clip(current_ab + chroma_shift, 0.0, 255.0)
    
    lab_lattice_u8 = np.clip(lab_lattice, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    graded_rgb = cv2.cvtColor(lab_lattice_u8, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32) / 255.0
    
    # Skin tone protection
    if protect_skin:
        skin_probs = compute_skin_mask_vectorized(rgb_lattice)
        blend_mask = (skin_probs * 0.85)[:, None]
        graded_rgb = graded_rgb * (1.0 - blend_mask) + rgb_lattice * blend_mask
        
    final_rgb = rgb_lattice * (1.0 - intensity) + graded_rgb * intensity
    final_rgb = np.clip(final_rgb, 0.0, 1.0)
    
    with open(output_cube_path, 'w') as f:
        f.write('TITLE "CineGrade Pro Cinematic Reference LUT"\n')
        f.write(f'LUT_3D_SIZE {lut_size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
        for i in range(len(final_rgb)):
            f.write(f'{final_rgb[i, 0]:.6f} {final_rgb[i, 1]:.6f} {final_rgb[i, 2]:.6f}\n')

def generate_auto_grade_lut(target_np, output_cube_path, lut_size=33, intensity=1.0, style="blockbuster"):
    """
    Hollywood Film Print Emulation Engine (Kodak 2383 / Fuji 3513 inspired):
    - Filmic S-curve with soft toe and gentle highlight shoulder roll-off
    - Signature cinema split-toning (Teal/Orange, Golden Hour, Noir, Clean)
    - Melanin skin-tone locus protection
    - Anchored blacks and whites for clean broadcast compliance
    """
    b_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    g_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    r_vals = np.linspace(0, 1, lut_size, dtype=np.float32)
    
    B, G, R = np.meshgrid(b_vals, g_vals, r_vals, indexing='ij')
    rgb = np.stack([R, G, B], axis=-1).reshape(-1, 3) # Shape: (N, 3)
    
    # 1. Kodak 2383 Filmic Tone Curve (Smooth S-Curve with protected highlights)
    x = rgb
    # Filmic curve function
    gamma_boost = 1.08
    x_g = np.power(x, gamma_boost)
    filmic = (x_g * (2.45 * x_g + 0.05)) / (x_g * (2.40 * x_g + 0.60) + 0.12)
    filmic = np.clip(filmic, 0.0, 1.0)
    
    # 2. Style-Specific Split-Toning & Chromatic Personality
    lum = 0.2126 * filmic[:, 0] + 0.7152 * filmic[:, 1] + 0.0722 * filmic[:, 2]
    
    if style == "golden_hour":
        # Warm golden highlight glow with rich amber midtones & soft film shadows
        s_curve = filmic * 0.75 + x * 0.25
        sh_w = np.clip((0.45 - lum) / 0.45, 0.0, 1.0)[:, None] ** 1.5
        hi_w = np.clip((lum - 0.40) / 0.60, 0.0, 1.0)[:, None] ** 1.5
        sh_tint = np.array([0.03, 0.015, -0.04], dtype=np.float32)
        hi_tint = np.array([0.09, 0.04, -0.07], dtype=np.float32)
        vib_boost = 0.18
    elif style == "noir":
        # Moody high-contrast with cool slate shadows and muted film tones
        s_curve = filmic * 0.90 + x * 0.10
        sh_w = np.clip((0.50 - lum) / 0.50, 0.0, 1.0)[:, None] ** 1.5
        hi_w = np.clip((lum - 0.55) / 0.45, 0.0, 1.0)[:, None] ** 1.5
        sh_tint = np.array([-0.04, -0.01, 0.05], dtype=np.float32)
        hi_tint = np.array([0.01, 0.01, -0.01], dtype=np.float32)
        vib_boost = -0.15
    elif style == "clean":
        # Crisp true-to-life broadcast color with extended dynamic range
        s_curve = filmic * 0.50 + x * 0.50
        sh_w = np.clip((0.35 - lum) / 0.35, 0.0, 1.0)[:, None] ** 1.5
        hi_w = np.clip((lum - 0.65) / 0.35, 0.0, 1.0)[:, None] ** 1.5
        sh_tint = np.array([-0.01, 0.01, 0.02], dtype=np.float32)
        hi_tint = np.array([0.01, 0.01, -0.01], dtype=np.float32)
        vib_boost = 0.22
    else: # "blockbuster" Hollywood standard
        # Hollywood Teal & Orange complementary contrast
        s_curve = filmic * 0.70 + x * 0.30
        sh_w = np.clip((0.45 - lum) / 0.45, 0.0, 1.0)[:, None] ** 1.6
        hi_w = np.clip((lum - 0.50) / 0.50, 0.0, 1.0)[:, None] ** 1.6
        # Rich teal in shadows, warm golden peach in highlights
        sh_tint = np.array([-0.05, 0.02, 0.07], dtype=np.float32)
        hi_tint = np.array([0.06, 0.025, -0.04], dtype=np.float32)
        vib_boost = 0.20
    
    graded = s_curve + sh_w * sh_tint + hi_w * hi_tint
    graded = np.clip(graded, 0.0, 1.0)
    
    # 3. Smart Vibrance (boosts muted tones while protecting saturated colors)
    max_c = np.max(graded, axis=1)
    min_c = np.min(graded, axis=1)
    sat = (max_c - min_c) / (max_c + 1e-6)
    vibrance_mult = (1.0 + vib_boost * (1.0 - sat))[:, None]
    
    lum_g = (0.2126 * graded[:, 0] + 0.7152 * graded[:, 1] + 0.0722 * graded[:, 2])[:, None]
    graded_vib = np.clip(lum_g + (graded - lum_g) * vibrance_mult, 0.0, 1.0)
    
    # 4. Skin Tone Preservation Gating
    skin_probs = compute_skin_mask_vectorized(rgb)
    skin_blend = (skin_probs * 0.75)[:, None]
    # Skin receives smooth warm film tone without turning orange/green
    skin_target = s_curve * 0.85 + rgb * 0.15
    rgb_final_graded = graded_vib * (1.0 - skin_blend) + skin_target * skin_blend
    
    # 5. Intensity blend & Output
    final_rgb = rgb * (1.0 - intensity) + rgb_final_graded * intensity
    final_rgb = np.clip(final_rgb, 0.0, 1.0)
    
    with open(output_cube_path, 'w') as f:
        f.write('TITLE "CineGrade AI Hollywood Film LUT"\n')
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

def run_grading_task(uid, ref_path, target_path, is_video, steps, size, ncc, output_cube, intensity=1.0, protect_skin=True, mode="reference", style="blockbuster"):
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
                generate_auto_grade_lut(target_thumb, output_cube, intensity=intensity, style=style)
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
            
            # Robust frame extraction (handles 4K/8K, HEVC, ProRes, BRAW, R3D, MKV, VFR)
            frame_rgb = None
            frame_jpg = os.path.join(tempfile.gettempdir(), f"frame_{uid}.jpg")
            try:
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-ss", "00:00:01",
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
                
            if frame_rgb is None:
                try:
                    cap = cv2.VideoCapture(target_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
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
                generate_auto_grade_lut(frame_rgb, output_cube, intensity=intensity, style=style)
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
    style: str = Form("blockbuster"), # "blockbuster", "golden_hour", "noir", "clean"
    intensity: float = Form(1.0),
    protect_skin: bool = Form(True),
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
        style
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
