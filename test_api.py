import requests
import sys

def test_api():
    url = "http://127.0.0.1:8000/api/grade"
    
    try:
        print("Sending POST request to /api/grade...")
        with open("cinematic_references/Amelie_1.jpg", "rb") as ref_file:
            # Create a dummy target image
            with open("cinematic_references/Amelie_2.jpg", "rb") as target_file:
                files = {
                    "reference": ("Amelie_1.jpg", ref_file, "image/jpeg"),
                    "target": ("Amelie_2.jpg", target_file, "image/jpeg")
                }
                data = {
                    "steps": 1,
                    "size": 512,
                    "ncc": "true"
                }
                
                response = requests.post(url, files=files, data=data, timeout=120)
                
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
