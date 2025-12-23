import pandas as pd
import zipfile
import os

csv_file = 'online_retail_II.csv'
zip_name = 'online_retail_II.zip'

print(f"Compressing {csv_file} to {zip_name}...")

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(csv_file, arcname=csv_file)

print(f"Compressed! New file size: {os.path.getsize(zip_name) / 1024 / 1024:.2f} MB")
