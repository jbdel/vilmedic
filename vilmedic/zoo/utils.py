from gdown import download as gdownload
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import sys


def download(tmp_zip_file, file_id, unzip_dir):
    outfile = os.path.join("data", tmp_zip_file) + ".zip"
    if not os.path.exists(outfile):
        gdownload("https://drive.google.com/uc?id=" + file_id,
                  outfile,
                  quiet=False)

    print("Unzipping...")
    os.makedirs(unzip_dir, exist_ok=True)
    with zipfile.ZipFile(outfile, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            zf.extract(member, unzip_dir)

    os.remove(outfile)
