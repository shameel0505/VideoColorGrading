import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
from PIL import Image
from grading import Inference
import argparse
from types import SimpleNamespace
import tempfile
import uuid
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import Optional

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

print("Loading AI Models... This might take a minute...")
try:
    config_args = SimpleNamespace(config='configs/prompts/video_demo.yaml')
    grader = Inference(config=config_args.config)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    grader = None

@app.get("/api/library")
async def get_library():
    ref_dir = "cinematic_references"
    if not os.path.exists(ref_dir):
        return []
    
    images = []
    for f in sorted(os.listdir(ref_dir)):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            images.append({"name": f.replace('_', ' ').replace('.jpg', ''), "path": f"/api/library/{f}"})
    return images

@app.get("/api/library/{filename}")
async def serve_library_image(filename: str):
    file_path = os.path.join("cinematic_references", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")

tasks = {}

def run_grading_task(uid, ref_path, target_path, is_video, steps, size, ncc, output_cube):
    tasks[uid] = {"status": "processing"}
    try:
        reference_image = Image.open(ref_path).convert('RGB').resize((size, size))
        reference_image_np = np.array(reference_image)
        
        target_ext = os.path.splitext(target_path)[1].lower()
        
        def load_image_with_raw_support(filepath):
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.dng', '.cr2', '.nef', '.arw']:
                import rawpy
                with rawpy.imread(filepath) as raw:
                    rgb = raw.postprocess(use_camera_wb=True)
                return rgb
            else:
                return np.array(Image.open(filepath).convert('RGB'))

        if not is_video:
            target_image = load_image_with_raw_support(target_path)
            
            # Create a lightweight thumbnail to generate the LUT (Zero RAM overhead)
            target_thumb = np.array(Image.fromarray(target_image).resize((512, 512), Image.Resampling.LANCZOS))
            
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
            
            # Apply the ultra-high-fidelity 64^3 LUT back to the full resolution raw image
            from pillow_lut import load_cube_file
            lut = load_cube_file(output_cube)
            
            output_jpg = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.jpg")
            Image.fromarray(target_image).filter(lut).save(output_jpg, quality=100)
            
            tasks[uid] = {"status": "completed", "result": {"output_media": output_jpg, "output_lut": output_cube, "type": "image"}}
        else:
            output_mp4 = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.mp4")
            
            import cv2
            import subprocess
            
            cap = cv2.VideoCapture(target_path)
            # Read a frame from the middle to get a good representative shot
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise Exception("Failed to read video file.")
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
            
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", target_path,
                "-vf", f"lut3d={output_cube}",
                "-c:a", "copy",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_mp4
            ]
            
            process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(f"FFmpeg failed: {process.stderr.decode('utf-8')}")
                
            tasks[uid] = {"status": "completed", "result": {"output_media": output_mp4, "output_lut": output_cube, "type": "video"}}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        tasks[uid] = {"status": "error", "error": str(e)}


from fastapi import BackgroundTasks

@app.post("/api/grade")
def process_grading(
    background_tasks: BackgroundTasks,
    reference: UploadFile = File(...),
    target: UploadFile = File(...),
    steps: int = Form(25),
    size: int = Form(512),
    ncc: bool = Form(True)
):
    if grader is None:
        raise HTTPException(status_code=500, detail="Models failed to load.")
        
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
    is_video = target_ext in ['.mp4', '.mov', '.avi']

    background_tasks.add_task(run_grading_task, uid, ref_path, target_path, is_video, steps, size, ncc, output_cube)
    
    return JSONResponse({"task_id": uid, "status": "processing"})

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(tasks[task_id])

@app.get("/api/download")
async def download_file(path: str):
    if os.path.exists(path):
        filename = os.path.basename(path)
        return FileResponse(path, filename=filename, media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
