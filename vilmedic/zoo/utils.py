import gdown
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import sys


def download(tmp_zip_file, file_id, unzip_dir, is_folder=False):
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir, exist_ok=True)
        if is_folder:
            gdown.download_folder("https://drive.google.com/drive/folders/" + file_id,
                                  os.path.split(unzip_dir)[0],
                                  quiet=False)
        else:
            outfile = os.path.join("data", tmp_zip_file) + ".zip"
            gdown.download("https://drive.google.com/uc?id=" + file_id,
                           outfile,
                           quiet=False)
            print("Unzipping...")
            with zipfile.ZipFile(outfile, 'r') as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    zf.extract(member, unzip_dir)

            os.remove(outfile)
