import lpips
import torch
import torch.nn.functional as F
import piq
import numpy as np
import imquality.brisque as brisque
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def calculate_ssim(gt_video, output_video):

    ssim_values = []
    for gt_frame, output_frame in zip(gt_video, output_video):
        gt_frame_np = np.array(gt_frame.resize(output_frame.size).convert('RGB'))
        output_frame_np = np.array(output_frame.convert('RGB'))
        ssim = compare_ssim(gt_frame_np, output_frame_np, win_size=5, channel_axis=-1)
        ssim_values.append(ssim)

    return np.mean(ssim_values)

def calculate_psnr(gt_video, output_video):

    psnr_values = []
    for gt_frame, output_frame in zip(gt_video, output_video):
        gt_frame_np = np.array(gt_frame.resize(output_frame.size).convert('RGB'))
        output_frame_np = np.array(output_frame.convert('RGB'))
        psnr = compare_psnr(gt_frame_np, output_frame_np)
        psnr_values.append(psnr)

    return np.mean(psnr_values)

def calculate_lpips_batch(loss_fn, gt_video, output_video, batch_size):

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    lpips_values = []

    tensors_a = [lpips.im2tensor(np.array(frame.resize(output_video[0].size))).to(device) for frame in gt_video]
    tensors_b = [lpips.im2tensor(np.array(frame.resize(output_video[0].size))).to(device) for frame in output_video]

    for i in range(0, len(tensors_a), batch_size):
        batch_a = torch.cat(tensors_a[i:i + batch_size], dim=0)
        batch_b = torch.cat(tensors_b[i:i + batch_size], dim=0)
        with torch.no_grad():
            batch_lpips = loss_fn(batch_a, batch_b)
        lpips_values.extend(batch_lpips.view(-1).cpu().numpy())

    return np.mean(lpips_values)

def calculate_blur(output_video):
    blurred_images = [np.array(frame.convert('L')) for frame in output_video] 
    blur_metrics = []
    for im in blurred_images:
        blur_score = measure.blur_effect(im, h_size=11)
        blur_metrics.append(blur_score)

    return np.mean(blur_metrics)

def calculate_brisque(output_video):
    brisque_scores = []
    for frame in output_video:
        frame_rgb = np.array(frame.convert('RGB')).transpose(2, 0, 1) / 255.0  # Convert to RGB and normalize
        frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).unsqueeze(0)
        score = piq.brisque(frame_tensor, data_range=1.0).item()
        brisque_scores.append(score)

    return np.mean(brisque_scores)

def evaluate(gt_video, output_video, ref_image):

    psnr = round(calculate_psnr(gt_video, output_video), 4)

    ssim = round(calculate_ssim(gt_video, output_video), 4)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
    lpip = round(calculate_lpips_batch(lpips_loss_fn, gt_video, output_video, 20), 4)

    brisque = round(calculate_brisque(output_video), 4)

    blur = round(calculate_blur(output_video), 4)

    return psnr, ssim, lpip, brisque, blur