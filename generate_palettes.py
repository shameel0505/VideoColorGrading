import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_colors(image, num_colors=5):
    # Resize for faster clustering
    small_image = cv2.resize(image, (100, 100))
    # Convert to RGB (OpenCV loads in BGR)
    img_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    # Sort colors by luminance (dark to light)
    colors = sorted(colors, key=lambda c: 0.299*c[0] + 0.587*c[1] + 0.114*c[2])
    return np.array(colors, dtype=int)

def append_palette(image_path, num_colors=5):
    img = cv2.imread(image_path)
    if img is None:
        return
        
    colors = extract_colors(img, num_colors)
    
    # Create a palette strip (height = 15% of image height, or min 50px)
    h, w, _ = img.shape
    palette_h = max(50, int(h * 0.15))
    palette_strip = np.zeros((palette_h, w, 3), dtype=np.uint8)
    
    swatch_w = w // num_colors
    for i, color in enumerate(colors):
        # Convert RGB back to BGR for OpenCV
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        start_x = i * swatch_w
        end_x = start_x + swatch_w if i < num_colors - 1 else w
        palette_strip[:, start_x:end_x] = color_bgr
        
    # Stack image and palette vertically
    new_img = np.vstack((img, palette_strip))
    cv2.imwrite(image_path, new_img)
    print(f"Palette appended to {os.path.basename(image_path)}")

if __name__ == "__main__":
    ref_dir = "cinematic_references"
    for f in os.listdir(ref_dir):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(ref_dir, f)
            append_palette(filepath)
