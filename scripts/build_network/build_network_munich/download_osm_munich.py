import os
import requests
import time

# ==============================
# 1. Set the save path
# ==============================
# Current script path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_DIR = os.path.join(BASE_DIR, "network")
FILE_NAME = "munich.osm"
# ==============================
# 2. bbox（A9Research section）
# ==============================
# min_lon, min_lat, max_lon, max_lat
BBOX = "11.35,48.05,11.80,48.30"

# ==============================
# 3. Overpass API URL
# ==============================
url = f"https://overpass-api.de/api/map?bbox={BBOX}"
for i in range(3):
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            print("Download success")
            break
        else:
            print(f"Failed, status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Request error: {e}")

    print("Retrying...")
    time.sleep(5)
# ==============================
# 4. Create a directory
# ==============================
os.makedirs(SAVE_DIR, exist_ok=True)

file_path = os.path.join(SAVE_DIR, FILE_NAME)

# ==============================
# 5. Download OSM data
# ==============================
print("Downloading OSM data...")

response = requests.get(url, stream=True)

if response.status_code == 200:
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    print("Download complete!")
    print(f"Saved to: {file_path}")

else:
    print("Failed to download data")
    print("Status code:", response.status_code)