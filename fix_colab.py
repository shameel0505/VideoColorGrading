import json

with open('CineGrade_Colab_Backend.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the setup cell
for cell in notebook['cells']:
    if cell['metadata'].get('id') == 'setup':
        source = cell['source']
        
        # Add the missing pip libraries
        for i, line in enumerate(source):
            if line.startswith('!pip install -q fastapi'):
                source[i] = '!pip install -q fastapi uvicorn pillow pillow-lut imageio imageio-ffmpeg rawpy python-multipart torch torchvision diffusers transformers accelerate omegaconf einops huggingface_hub\n'
        
        # Add the download script
        if '!python download_lite.py\n' not in source:
            source.insert(-1, '!python download_lite.py\n')
            source.insert(-1, 'print("\\n⏳ Downloading AI Neural Models (this takes a few minutes)...")\n')

with open('CineGrade_Colab_Backend.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)
    f.write('\n')
