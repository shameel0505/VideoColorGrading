import os
import requests
from bs4 import BeautifulSoup
import time
import re
import urllib.parse

MOVIES = [
    "Blade Runner 2049",
    "The Matrix",
    "Mad Max Fury Road",
    "Amelie",
    "Se7en",
    "The Grand Budapest Hotel",
    "Moonlight 2016",
    "Drive 2011",
    "Dune 2021",
    "The Batman 2022",
    "Oppenheimer",
    "Joker 2019",
    "Her 2013",
    "La La Land",
    "Arrival 2016",
    "Parasite",
    "Interstellar",
    "The Revenant",
    "John Wick",
    "Skyfall",
    "Prisoners",
    "In the Mood for Love",
    "No Country for Old Men",
    "Fargo"
]

OUTPUT_DIR = "cinematic_references"
os.makedirs(OUTPUT_DIR, exist_ok=True)
headers = {"User-Agent": "Mozilla/5.0"}

for movie in MOVIES:
    print(f"\n🔍 Searching for {movie} on Film-Grab...")
    try:
        # Search for the movie
        search_url = f"https://film-grab.com/?s={urllib.parse.quote_plus(movie)}"
        r = requests.get(search_url, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Get the first search result URL
        result_links = soup.select('h2.entry-title a')
        if not result_links:
            print(f"❌ No results found for {movie}")
            continue
            
        movie_url = result_links[0]['href']
        print(f"  -> Found post: {movie_url}")
        
        # Go to the movie post
        r2 = requests.get(movie_url, headers=headers)
        r2.raise_for_status()
        soup2 = BeautifulSoup(r2.text, 'html.parser')
        
        # Find all images
        imgs = soup2.find_all('img')
        saved_count = 0
        clean_name = movie.replace(' ', '_').replace(':', '')
        
        for img in imgs:
            if saved_count >= 3:
                break
                
            src = img.get('src')
            if not src or 'wp-content/uploads' not in src:
                continue
            
            # Film-grab thumbnails have dimensions like -150x150 in the filename.
            # We remove this to get the full resolution image!
            full_src = re.sub(r'-\d+x\d+(?=\.\w+)', '', src)
            
            try:
                img_data = requests.get(full_src, headers=headers, timeout=10).content
                ext = os.path.splitext(full_src)[1]
                if not ext: ext = ".jpg"
                filename = f"{clean_name}_{saved_count+1}{ext}"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                print(f"  ✅ Saved: {filename}")
                saved_count += 1
            except Exception as e:
                print(f"  ❌ Failed to download {full_src}: {e}")
                
    except Exception as e:
        print(f"❌ Failed to scrape {movie}: {e}")
        
    time.sleep(1) # Be polite

print("\n🎉 All downloads complete!")
