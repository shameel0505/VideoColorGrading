import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from grading import Inference
import argparse
from types import SimpleNamespace
import tempfile
import uuid
import rawpy

def load_raw_image(file_obj):
    with rawpy.imread(file_obj.name) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
    return rgb

import torch
torch.set_grad_enabled(False)

# Initialize the model globally to avoid reloading on every request
try:
    print("Loading AI Models... This might take a minute...")
    config_args = SimpleNamespace(config='configs/prompts/video_demo.yaml')
    grader = Inference(config=config_args.config)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    grader = None

def convert_image_to_video(image_path, output_path):
    """Converts a single image to a 1-frame mp4 video."""
    img = cv2.imread(image_path)
    height, width, layers = img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 1, (width, height))
    video.write(img)
    video.release()
    return output_path

def convert_video_to_image(video_path, output_path):
    """Extracts the first frame of a video to an image."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()
    return output_path

def color_grade_handler(ref_image, target_video, target_image, steps, size, ncc):
    if not grader:
        return None, None, "Error: AI Models failed to load. Check console logs."

    if ref_image is None:
        return None, None, "Please provide a reference image."

    if target_video is None and target_image is None:
        return None, None, "Please provide either a target video or a target image."

    if target_video is not None and target_image is not None:
        return None, None, "Please provide ONLY a target video OR a target image, not both."

    # Save reference image
    ref_path = tempfile.mktemp(suffix=".jpg")
    ref_pil = Image.fromarray(ref_image)
    ref_pil.save(ref_path)

    # Unique output path
    uid = str(uuid.uuid4())[:8]
    output_mp4 = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.mp4")
    output_cube = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.cube")

    # Read reference for model
    reference_image = Image.open(ref_path).convert('RGB').resize((size, size))
    reference_image_np = np.array(reference_image)

    print(f"Processing with Seed: 42, Steps: {steps}, Size: {size}, NCC: {ncc}")
    
    try:
        if target_image is not None:
            # Strip alpha channel if image is RGBA (e.g. PNGs)
            if target_image.shape[-1] == 4:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_RGBA2RGB)
                
            # Single Image path - bypass lossy video compression to preserve RAW quality
            input_frames = [target_image]
            result_frames = grader(
                ref_sequence=reference_image_np,
                input_frames=input_frames,
                return_frames=True,
                save_lut_path=output_cube,
                random_seed=42, 
                step=steps, 
                size=size, 
                ncc=ncc
            )
            output_jpg = os.path.join(tempfile.gettempdir(), f"graded_output_{uid}.jpg")
            Image.fromarray(result_frames[0]).save(output_jpg, quality=100)
            return output_jpg, output_cube, "Success! Color graded high-res image generated."
        else:
            # Video path
            result_path = grader(
                ref_sequence=reference_image_np, 
                input_path=target_video, 
                save_path=output_mp4, 
                save_lut_path=output_cube,
                random_seed=42, 
                step=steps, 
                size=size, 
                ncc=ncc
            )
            return result_path, output_cube, "Success! Color graded video generated."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error during processing: {str(e)}"

def get_gallery_images():
    ref_dir = "cinematic_references"
    if not os.path.exists(ref_dir):
        return []
    
    images = []
    # Sort files to ensure grouped movies
    for f in sorted(os.listdir(ref_dir)):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(ref_dir, f))
    return images

# Define Gradio UI with a professional Dark Theme
with gr.Blocks(title="AI Cinematic Color Grading Studio", theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# 🎬 AI Cinematic Color Grading Studio")
    gr.Markdown("Transform your footage into a cinematic masterpiece using cutting-edge neural models.")
    
    with gr.Tabs():
        with gr.TabItem("Step 1: Choose Your Style"):
            gr.Markdown("### Select a Professional Cinematic Palette")
            gr.Markdown("Click on any of the iconic cinematic stills below to load its color palette, or upload your own reference manually.")
            
            gallery = gr.Gallery(
                value=get_gallery_images,
                label="Cinematic Reference Library",
                show_label=True,
                elem_id="gallery",
                columns=[5],
                rows=[5],
                object_fit="contain",
                height="auto"
            )
            
            with gr.Row():
                ref_in = gr.Image(label="Selected Reference Image", type="numpy", interactive=True)
                ref_dng = gr.File(label="...OR Upload RAW/DNG Reference (.dng, .cr2, etc.)")
            
            def on_gallery_select(evt: gr.SelectData):
                # Safely load the image path depending on Gradio version's SelectData structure
                if isinstance(evt.value, str):
                    img_path = evt.value
                elif evt.value and 'name' in evt.value:
                    img_path = evt.value['name']
                elif evt.value and 'image' in evt.value and 'path' in evt.value['image']:
                    img_path = evt.value['image']['path']
                else:
                    # Fallback to reconstructing the path if we have the index
                    images = get_gallery_images()
                    img_path = images[evt.index]
                    
                return np.array(Image.open(img_path).convert('RGB'))
                
            gallery.select(fn=on_gallery_select, inputs=[], outputs=[ref_in])
            
        with gr.TabItem("Step 2: Upload Target"):
            gr.Markdown("### Upload the video or photo you want to color grade")
            with gr.Row():
                target_vid = gr.Video(label="Target Video (.mp4, .mov)")
                target_img = gr.Image(label="Target Image (JPG/PNG)", type="numpy")
            target_dng = gr.File(label="...OR Upload RAW Target (.dng, .cr2)")

        with gr.TabItem("Step 3: Render Masterpiece"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Engine Settings")
                    steps_slider = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="Inference Steps (Higher = Better Quality)")
                    size_slider = gr.Slider(minimum=512, maximum=512, value=512, step=64, label="Resolution Size (Fixed)")
                    ncc_checkbox = gr.Checkbox(label="Enable Neural Color Correction (NCC)", value=True)
                    submit_btn = gr.Button("🎨 Render Masterpiece", variant="primary", size="lg")
                    status_out = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### Graded Output")
                    out_vid = gr.Video(label="Result Video", visible=True)
                    out_img = gr.Image(label="Result Image", visible=False)
                    out_lut = gr.File(label="Download 3D LUT (.cube)")

    def route_output(ref, ref_dng, vid, img, target_dng, steps, size, ncc):
        if ref is None and ref_dng is None:
            return gr.update(value=None, visible=True), gr.update(value=None, visible=False), gr.update(value=None), "Please select a reference image from the gallery or upload one."
            
        if ref_dng is not None:
            try:
                ref = load_raw_image(ref_dng)
            except Exception as e:
                return gr.update(value=None, visible=True), gr.update(value=None, visible=False), gr.update(value=None), f"Failed to load RAW reference: {e}"

        if target_dng is not None:
            try:
                img = load_raw_image(target_dng)
            except Exception as e:
                return gr.update(value=None, visible=True), gr.update(value=None, visible=False), gr.update(value=None), f"Failed to load RAW target: {e}"

        result_file, result_lut, status = color_grade_handler(ref, vid, img, steps, size, ncc)
        if result_file and result_file.endswith('.jpg'):
            return gr.update(value=None, visible=False), gr.update(value=result_file, visible=True), gr.update(value=result_lut), status
        elif result_file and result_file.endswith('.mp4'):
            return gr.update(value=result_file, visible=True), gr.update(value=None, visible=False), gr.update(value=result_lut), status
        else:
            return gr.update(value=None, visible=True), gr.update(value=None, visible=False), gr.update(value=None), status

    submit_btn.click(
        route_output,
        inputs=[ref_in, ref_dng, target_vid, target_img, target_dng, steps_slider, size_slider, ncc_checkbox],
        outputs=[out_vid, out_img, out_lut, status_out]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
