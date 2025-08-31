import os
import zipfile
import subprocess

# Upload kaggle api to this folder
def download_asl_dataset(
    kaggle_json_path="kaggle.json",
    dataset="grassknoted/asl-alphabet",
    download_dir="../data/raw",
    extract_dir="../data/asl_alphabet"
):
    """
    Downloads and extracts the ASL Alphabet dataset from Kaggle.

    Args:
        kaggle_json_path (str): Path to kaggle.json file with API credentials.
        dataset (str): Kaggle dataset identifier.
        download_dir (str): Directory where dataset zip will be downloaded.
        extract_dir (str): Directory where dataset will be extracted.

    Returns:
        str: Path to extracted dataset directory.
    """
    # Ensure Kaggle API key exists
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(
            f"{kaggle_json_path} not found. Please place your Kaggle API key in the project root."
        )

    # Setup Kaggle API credentials
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    subprocess.run(["cp", kaggle_json_path, os.path.expanduser("~/.kaggle/")], check=True)
    subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")], check=True)

    os.makedirs(download_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", download_dir],
        check=True
    )

    dataset_zip = os.path.join(download_dir, dataset.split("/")[-1] + ".zip")
    print("📂 Extracting dataset...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir


if __name__ == "__main__":
    dataset_path = download_asl_dataset()
    print("Dataset ready at:", dataset_path)