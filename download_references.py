import os
import shutil
from bing_image_downloader import downloader

MOVIES = [
    "Blade Runner 2049 movie still site:film-grab.com",
    "The Matrix 1999 movie still site:film-grab.com",
    "Mad Max Fury Road movie still site:film-grab.com",
    "Amelie movie still site:film-grab.com",
    "Se7en movie still site:film-grab.com",
    "The Grand Budapest Hotel movie still site:film-grab.com",
    "Moonlight 2016 movie still site:film-grab.com",
    "Drive 2011 movie still site:film-grab.com"
]

OUTPUT_DIR = "cinematic_references"
TEMP_DIR = "temp_bing_downloads"

def download_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Initializing Bing Auto-Downloader to ./{OUTPUT_DIR}...")
    
    for movie in MOVIES:
        print(f"\n🔍 Downloading: {movie}")
        try:
            # Download to a temporary structured folder
            downloader.download(
                movie, 
                limit=3, 
                output_dir=TEMP_DIR, 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=10, 
                verbose=False
            )
            
            # The downloader replaces special characters like ':' with '_'
            folder_name = movie.replace(":", "_")
            query_folder = os.path.join(TEMP_DIR, folder_name)
            if os.path.exists(query_folder):
                images = os.listdir(query_folder)
                # Clean filename
                clean_name = "".join([c if c.isalnum() else "_" for c in movie.split("movie")[0].strip()])
                
                for i, img_file in enumerate(images):
                    src_path = os.path.join(query_folder, img_file)
                    ext = os.path.splitext(img_file)[1]
                    dest_path = os.path.join(OUTPUT_DIR, f"{clean_name}_{i+1}{ext}")
                    
                    # Move image to final directory
                    shutil.move(src_path, dest_path)
                    print(f"  ✅ Saved: {dest_path}")
                    
        except Exception as e:
            print(f"  ❌ Failed to download {movie}: {e}")

    # Cleanup temp dir
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        
    print("\n🎉 All downloads complete! Check your 'cinematic_references' folder.")

if __name__ == "__main__":
    download_images()
