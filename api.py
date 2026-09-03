import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
from PIL import Image
import argparse
from types import SimpleNamespace
import tempfile
import uuid
import torch
import shutil
import subprocess
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import imageio_ffmpeg
from pillow_lut import load_cube_file

torch.set_grad_enabled(False)

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

def compute_skin_mask(rgb_np):
    """Generates a soft probability mask (0.0 to 1.0) of human skin tones in YCbCr space."""
    ycrcb = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    
    # Skin locus approximately centered at Cr ~ 150, Cb ~ 105
    skin_cr = np.exp(-((cr - 150.0) ** 2) / (2.0 * (20.0 ** 2)))
    skin_cb = np.exp(-((cb - 105.0) ** 2) / (2.0 * (18.0 ** 2)))
    mask = skin_cr * skin_cb
    
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0
    lum_gate = np.clip((y - 0.12) / 0.15, 0.0, 1.0) * np.clip((0.95 - y) / 0.15, 0.0, 1.0)
    return np.clip(mask * lum_gate, 0.0, 1.0)

def generate_pro_reference_lut(ref_np, target_np, output_cube_path, lut_size=33, intensity=1.0, protect_skin=True):
    """
    Generates a professional 3D Look-Up Table (.cube) from reference scene
    with Zone-Aware (Shadows/Midtones/Highlights) transfer and Skin-Tone preservation.
    """
    ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    ref_l = ref_lab[:, :, 0]
    tgt_l = tgt_lab[:, :, 0]
    
    ref_mean = np.mean(ref_lab, axis=(0, 1))
    ref_std = np.std(ref_lab, axis=(0, 1)) + 1e-5
    tgt_mean = np.mean(tgt_lab, axis=(0, 1))
    tgt_std = np.std(tgt_lab, axis=(0, 1)) + 1e-5
    
    # Zone statistics: Shadows (L < 85), Midtones (85 <= L < 170), Highlights (L >= 170)
    shadow_mask_ref = ref_l < 85
    highlight_mask_ref = ref_l >= 170
    mid_mask_ref = ~shadow_mask_ref & ~highlight_mask_ref
    
    ref_ab_shadow = np.mean(ref_lab[shadow_mask_ref, 1:3], axis=0) if np.any(shadow_mask_ref) else ref_mean[1:3]
    ref_ab_mid = np.mean(ref_lab[mid_mask_ref, 1:3], axis=0) if np.any(mid_mask_ref) else ref_mean[1:3]
    ref_ab_high = np.mean(ref_lab[highlight_mask_ref, 1:3], axis=0) if np.any(highlight_mask_ref) else ref_mean[1:3]

    with open(output_cube_path, 'w') as f:
        f.write('TITLE "CineGrade AI Pro Reference LUT"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    r_norm = r / (lut_size - 1)
                    g_norm = g / (lut_size - 1)
                    b_norm = b / (lut_size - 1)
                    
                    rgb_orig = np.array([[[r_norm * 255.0, g_norm * 255.0, b_norm * 255.0]]], dtype=np.uint8)
                    lab = cv2.cvtColor(rgb_orig, cv2.COLOR_RGB2LAB).astype(np.float32)
                    
                    # Luminance alignment with soft roll-off
                    l_val = lab[0, 0, 0]
                    norm_l = l_val / 255.0
                    scaled_l = (l_val - tgt_mean[0]) * (ref_std[0] / tgt_std[0]) * 0.75 + ref_mean[0]
                    lab[0, 0, 0] = np.clip(scaled_l, 0.0, 255.0)
                    
                    # Zone-based chroma interpolation
                    if norm_l < 0.33:
                        t = norm_l / 0.33
                        target_ab = ref_ab_shadow * (1.0 - t) + ref_ab_mid * t
                    else:
                        t = (norm_l - 0.33) / 0.67
                        target_ab = ref_ab_mid * (1.0 - t) + ref_ab_high * t
                        
                    # Statistical Chroma Transfer
                    trans_a = (lab[0, 0, 1] - tgt_mean[1]) * (ref_std[1] / tgt_std[1]) * 0.8 + target_ab[0]
                    trans_b = (lab[0, 0, 2] - tgt_mean[2]) * (ref_std[2] / tgt_std[2]) * 0.8 + target_ab[1]
                    
                    lab[0, 0, 1] = np.clip(trans_a, 0.0, 255.0)
                    lab[0, 0, 2] = np.clip(trans_b, 0.0, 255.0)
                    
                    # Convert back to RGB
                    graded_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
                    orig_rgb_flt = np.array([r_norm, g_norm, b_norm], dtype=np.float32)
                    
                    # Skin tone preservation
                    if protect_skin:
                        sample_rgb = (orig_rgb_flt * 255).astype(np.uint8).reshape(1, 1, 3)
                        skin_prob = compute_skin_mask(sample_rgb)[0, 0]
                        if skin_prob > 0.05:
                            blend_factor = skin_prob * 0.75
                            graded_rgb[0, 0] = graded_rgb[0, 0] * (1.0 - blend_factor) + orig_rgb_flt * blend_factor
                    
                    # Final blend by intensity
                    final_rgb = orig_rgb_flt * (1.0 - intensity) + graded_rgb[0, 0] * intensity
                    final_rgb = np.clip(final_rgb, 0.0, 1.0)
                    
                    f.write(f"{final_rgb[0]:.6f} {final_rgb[1]:.6f} {final_rgb[2]:.6f}\n")

def generate_auto_grade_lut(target_np, output_cube_path, lut_size=33, intensity=1.0):
    """
    Autonomous Pro Colorist Engine:
    1. Dynamic Gray-World Chromatic Adaptation (Auto White Balance).
    2. Dynamic Range Stretch with soft Hermite highlight & shadow roll-offs.
    3. Filmic S-Curve contrast and smart memory-color vibrance enhancement.
    """
    img_flt = target_np.astype(np.float32) / 255.0
    
    # 1. White Balance estimation via Shades of Gray (p=6 norm)
    p = 6.0
    minkowski_norm = np.power(np.mean(np.power(img_flt, p), axis=(0, 1)), 1.0 / p) + 1e-6
    gray_target = np.mean(minkowski_norm)
    wb_gain = gray_target / minkowski_norm
    wb_gain = np.clip(wb_gain, 0.82, 1.22)
    
    # 2. Dynamic range percentiles
    low_p = np.percentile(img_flt, 1.0)
    high_p = np.percentile(img_flt, 99.0)
    if high_p - low_p < 0.1:
        low_p = 0.0
        high_p = 1.0
        
    with open(output_cube_path, 'w') as f:
        f.write('TITLE "CineGrade AI Auto Pro Grade"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    r_norm = r / (lut_size - 1)
                    g_norm = g / (lut_size - 1)
                    b_norm = b / (lut_size - 1)
                    
                    rgb = np.array([r_norm, g_norm, b_norm], dtype=np.float32)
                    
                    # 1. White Balance
                    rgb_wb = rgb * wb_gain
                    
                    # 2. Dynamic Range Stretch
                    rgb_stretched = (rgb_wb - low_p) / (high_p - low_p + 1e-6)
                    rgb_stretched = np.clip(rgb_stretched, 0.0, 1.0)
                    
                    # 3. Filmic S-Curve Tone Mapping
                    c = rgb_stretched
                    s_curve = c * c * (3.0 - 2.0 * c)
                    rgb_toned = c * 0.35 + s_curve * 0.65
                    
                    # 4. Vibrance & Skin Tone Protection
                    max_c = np.max(rgb_toned)
                    min_c = np.min(rgb_toned)
                    sat = (max_c - min_c) / (max_c + 1e-6)
                    vibrance_boost = 1.0 + 0.22 * (1.0 - sat)
                    luma = 0.2126 * rgb_toned[0] + 0.7152 * rgb_toned[1] + 0.0722 * rgb_toned[2]
                    rgb_vib = luma + (rgb_toned - luma) * vibrance_boost
                    
                    # Blend by intensity
                    final_rgb = rgb * (1.0 - intensity) + rgb_vib * intensity
                    final_rgb = np.clip(final_rgb, 0.0, 1.0)
                    
                    f.write(f"{final_rgb[0]:.6f} {final_rgb[1]:.6f} {final_rgb[2]:.6f}\n")

def run_grading_task(uid, ref_path, target_path, is_video, steps, size, ncc, output_cube, intensity=1.0, protect_skin=True, mode="reference"):
    tasks[uid] = {"status": "processing"}
    try:
        def load_image_with_raw_support(filepath):
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.dng', '.cr2', '.nef', '.arw']:
                import rawpy
                with rawpy.imread(filepath) as raw:
                    return raw.postprocess(use_camera_wb=True)
            else:
                return np.array(Image.open(filepath).convert('RGB'))

        if not is_video:
            target_image = load_image_with_raw_support(target_path)
            target_thumb = np.array(Image.fromarray(target_image).resize((512, 512), Image.Resampling.LANCZOS))
            
            if mode == "auto":
                generate_auto_grade_lut(target_thumb, output_cube, intensity=intensity)
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
            
            cap = cv2.VideoCapture(target_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise Exception("Failed to read target video file.")
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if mode == "auto":
                generate_auto_grade_lut(frame_rgb, output_cube, intensity=intensity)
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
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            escaped_cube = output_cube.replace("\\", "/").replace(":", "\\:")
            ffmpeg_cmd = [
                ffmpeg_exe, "-y", "-i", target_path,
                "-vf", f"lut3d={escaped_cube}",
                "-c:a", "copy",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_mp4
            ]
            
            process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.returncode != 0:
                tasks[uid] = {"status": "error", "error": f"FFmpeg failed: {process.stderr.decode('utf-8', errors='ignore')}"}
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

@app.post("/api/grade")
def process_grading(
    background_tasks: BackgroundTasks,
    target: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    mode: str = Form("reference"), # "reference" or "auto"
    intensity: float = Form(1.0),
    protect_skin: bool = Form(True),
    steps: int = Form(25),
    size: int = Form(512),
    ncc: bool = Form(True)
):
    uid = uuid.uuid4().hex[:8]
    
    ref_path = None
    if mode == "reference" and reference is not None:
        ref_ext = os.path.splitext(reference.filename)[1].lower()
        ref_path = os.path.join(tempfile.gettempdir(), f"ref_{uid}{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        
    target_ext = os.path.splitext(target.filename)[1].lower()
    target_path = os.path.join(tempfile.gettempdir(), f"target_{uid}{target_ext}")
    with open(target_path, "wb") as f:
        shutil.copyfileobj(target.file, f)
        
    output_cube = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.cube")
    is_video = target_ext in ['.mp4', '.mov', '.avi', '.mkv']
    
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
        mode
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
