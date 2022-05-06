import gdown
import os
import zipfile
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files


def edit_vocab_path_in_dict(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = edit_vocab_path_in_dict(v, key, replace_value)
    if key in obj:
        obj[key] = os.path.join(replace_value, obj[key])
    return obj


def download_images(data_name, file_id, unzip_dir):
    zip_name = os.path.join("data", data_name) + ".zip"
    target_dir = os.path.join(unzip_dir, data_name)

    if not os.path.exists(target_dir):
        gdown.download(url="https://drive.google.com/uc?id=" + file_id,
                       output=zip_name,
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
    # is HuggingFace repo
    if '/' in file_id:
        files = list(set(list_repo_files(repo_id=file_id)).difference({'README.md', '.gitattributes'}))
        for f in files:
            try:
                hf_hub_download(file_id, filename=f, cache_dir=unzip_dir, force_filename=f)
            except Exception as e:
                print(e)

    else:  # Otherwise gdrive
        gdown.download_folder(id=file_id,
                              output=unzip_dir,
                              quiet=False)


def download_data(file_id, unzip_dir):
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir, exist_ok=True)
    gdown.download_folder(id=file_id,
                          output=unzip_dir,
                          quiet=False)
