import io
import time
import zipfile
import requests
import shutil
import os


def download_math():
    from huggingface_hub import snapshot_download

    repo_id = "lighteval/MATH"
    local_dir = "./datasets/MATH"
    cache_dir = local_dir + "/.cache"
    while True:
        try:
            snapshot_download(
                cache_dir=cache_dir,
                local_dir=local_dir,
                repo_id=repo_id,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as e:
            time.sleep(5)
        else:
            print("Done")
            break


def download_gsm8k():
    from huggingface_hub import hf_hub_download

    repo_id = "openai/gsm8k"
    local_dir = "./datasets/gsm8k"
    hf_hub_download(repo_id, local_dir, repo_type="dataset")


def download_prm800k():
    repo_url = "https://github.com/openai/prm800k"
    local_dir = "./datasets"
    folder_name = "prm800k"
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"
    response = requests.get(zip_url)

    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file_info in z.infolist():
                if file_info.filename.startswith(
                    f"{repo_url.split('/')[-1]}-main/{folder_name}/"
                ):
                    z.extract(file_info, local_dir)
        print(f"Downloaded {zip_url}")

        extracted_folder = os.path.join(
            local_dir, f"{repo_url.split('/')[-1]}-main", folder_name
        )
        shutil.move(extracted_folder, local_dir)
        os.rmdir(os.path.join(local_dir, f"{repo_url.split('/')[-1]}-main"))

    else:
        print(f"Failed to download {zip_url}")


def download_collegemath():
    local_dir = "./datasets"
    folder_name = "collegemath"

    os.makedirs(os.path.join(local_dir, folder_name), exist_ok=True)
    files_to_download = [
        "https://github.com/microsoft/unilm/raw/refs/heads/master/mathscale/MWPBench/data/full_test.json",
        "https://github.com/microsoft/unilm/raw/refs/heads/master/mathscale/MWPBench/data/full_train.json",
    ]

    for url in files_to_download:
        filename = url.split("/")[-1].replace(".json", ".jsonl")
        file_path = os.path.join(local_dir, folder_name, filename)

        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}")


if __name__ == "__main__":
    # download_math()
    # download_gsm8k()
    download_prm800k()
    # download_collegemath()
