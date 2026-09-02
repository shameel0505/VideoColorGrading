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

app = FastAPI(title="CineGrade AI API")

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI models if available
grader = None
try:
    print("Checking AI Pipeline Models...")
    from grading import Inference
    config_args = SimpleNamespace(config='configs/prompts/video_demo.yaml')
    grader = Inference(config=config_args.config)
    print("AI Diffusion Models loaded successfully!")
except Exception as e:
    print(f"Notice: Running in direct 3D LUT Color Transfer mode ({e})")
    grader = None

@app.get("/api/library")
async def get_library():
    ref_dir = "cinematic_references"
    if not os.path.exists(ref_dir):
        return []
    
    images = []
    for f in sorted(os.listdir(ref_dir)):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            images.append({
                "name": f.replace('_', ' ').replace('.jpg', '').replace('.png', ''), 
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

def generate_color_transfer_lut(ref_np, target_np, output_cube_path, lut_size=33):
    """Generates an authentic Adobe 3D Look-Up Table (.cube) from reference scene."""
    ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    ref_mean = ref_lab.mean(axis=(0, 1))
    ref_std = ref_lab.std(axis=(0, 1)) + 1e-6
    tgt_mean = tgt_lab.mean(axis=(0, 1))
    tgt_std = tgt_lab.std(axis=(0, 1)) + 1e-6

    with open(output_cube_path, 'w') as f:
        f.write("TITLE \"CineGrade AI Reference LUT\"\n")
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    rgb = np.array([[[
                        r / (lut_size - 1) * 255.0, 
                        g / (lut_size - 1) * 255.0, 
                        b / (lut_size - 1) * 255.0
                    ]]], dtype=np.uint8)
                    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                    
                    # Statistical color transfer in perceptually uniform LAB space
                    lab[0, 0] = (lab[0, 0] - tgt_mean) * (ref_std / tgt_std) * 0.85 + ref_mean
                    lab[0, 0, 0] = np.clip(lab[0, 0, 0], 0, 255)
                    lab[0, 0, 1] = np.clip(lab[0, 0, 1], 0, 255)
                    lab[0, 0, 2] = np.clip(lab[0, 0, 2], 0, 255)
                    
                    graded_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
                    f.write(f"{graded_rgb[0, 0, 0]:.6f} {graded_rgb[0, 0, 1]:.6f} {graded_rgb[0, 0, 2]:.6f}\n")

def run_grading_task(uid, ref_path, target_path, is_video, steps, size, ncc, output_cube):
    tasks[uid] = {"status": "processing"}
    try:
        reference_image = Image.open(ref_path).convert('RGB').resize((size, size))
        reference_image_np = np.array(reference_image)
        
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
            
            if grader is not None:
                grader(
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
                generate_color_transfer_lut(reference_image_np, target_thumb, output_cube)
            
            # Apply LUT to image
            lut = load_cube_file(output_cube)
            output_jpg = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.jpg")
            Image.fromarray(target_image).filter(lut).save(output_jpg, quality=100)
            
            tasks[uid] = {
                "status": "completed", 
                "result": {
                    "output_media": output_jpg, 
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
            
            if grader is not None:
                grader(
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
                generate_color_transfer_lut(reference_image_np, frame_rgb, output_cube)
            
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
                # Fallback to OpenCV frame processing if filter fails
                tasks[uid] = {"status": "error", "error": f"FFmpeg failed: {process.stderr.decode('utf-8', errors='ignore')}"}
                return
                
            tasks[uid] = {
                "status": "completed", 
                "result": {
                    "output_media": output_mp4, 
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
    reference: UploadFile = File(...),
    target: UploadFile = File(...),
    steps: int = Form(25),
    size: int = Form(512),
    ncc: bool = Form(True)
):
    uid = uuid.uuid4().hex[:8]
    
    # Save uploaded files
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
        output_cube
    )
    
    return {"task_id": uid, "status": "processing"}

@app.get("/api/status/{task_id}")
def check_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.api_route("/api/download", methods=["GET", "HEAD"])
def download_by_query(path: str):
    # Support both full file path and basename
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
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
