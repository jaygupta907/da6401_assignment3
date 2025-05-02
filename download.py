import os
import urllib.request
import tarfile

def download_and_extract_dakshina(dest_dir="./datasets"):
    url = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
    tar_path = os.path.join(dest_dir, "dakshina_dataset_v1.0.tar")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(tar_path):
        print("Downloading Dakshina dataset...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    else:
        print("Dataset archive already exists.")

    print("Extracting dataset...")
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(path=dest_dir)
    print("Extraction complete.")

    os.remove(tar_path)
    print("Removed archive tar file.")


if __name__ == '__main__':
    download_and_extract_dakshina()