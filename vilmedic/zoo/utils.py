import gdown
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import sys


def download_data(data_name, file_id, unzip_dir):
    zip_name = os.path.join("data", data_name) + ".zip"
    target_dir = os.path.join(unzip_dir, data_name)

    if not os.path.exists(target_dir):
        gdown.download("https://drive.google.com/uc?id=" + file_id,
                       zip_name,
                       quiet=False)
        print("Unzipping...")
        with zipfile.ZipFile(zip_name, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting '):
                zf.extract(member, unzip_dir)

        os.remove(zip_name)
        return
    print('{} already exists'.format(target_dir))


def download_model(file_id, unzip_dir):
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir, exist_ok=True)
        gdown.download_folder("https://drive.google.com/drive/folders/" + file_id,
                              os.path.split(unzip_dir)[0],
                              quiet=False)
