import urllib.request
import zipfile
import os

url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

os.makedirs("data", exist_ok=True)

zip_path = "data/jena.zip"

print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting dataset...")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("data")

print("Done!")
print("Dataset saved in data/ folder.")