# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
Download the FungiCLEF 2023 challenge dataset based on the Danish Fungi 2020 and 2021 preprocessed images and metadata.

Downloads:
 - Training images (max side size 300px; DF20) [~6.5GB]
 - Validation + Public Test images (max side size 300px; DF21) [~2.5GB]
 - Training, Validation, and Public Test metadata CSVs
"""

import dataclasses
import os
import tarfile

import requests
import tqdm
import tyro

URLS = {
    "train_imgs": "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-300px.tar.gz",
    "val_imgs": "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF21_300px.tar.gz",
    "train_metadata": "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/FungiCLEF2023_train_metadata_PRODUCTION.csv",
    "val_metadata": "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/FungiCLEF2023_val_metadata_PRODUCTION.csv",
}


@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save downloaded archives and extract them."""
    chunk_size_kb: int = 1024
    """Download chunk size in KB."""
    download_train_imgs: bool = True
    """Whether to download training images (DF20) [~6.5GB]."""
    download_val_imgs: bool = True
    """Whether to download validation and public test images (DF21) [~2.5GB]."""
    download_train_metadata: bool = True
    """Whether to download training metadata CSV."""
    download_val_metadata: bool = True
    """Whether to download validation metadata CSV."""
    unzip: bool = True
    """Whether to extract downloaded archives."""


def download_file(name: str, url: str, dest_dir: str, chunk_size: int) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(url)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        print(f"{filename} already exists, skipping download")
        return dest_path
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with (
        open(dest_path, "wb") as f,
        tqdm.tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {filename}"
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    return dest_path


def extract_file(archive_path: str, dest_dir: str):
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tqdm.tqdm(
            tar, desc=f"Extracting {os.path.basename(archive_path)}"
        ):
            tar.extract(member, path=dest_dir)


def main(args: Args):
    base_dir = args.dir
    chunk_size = args.chunk_size_kb * 1024
    for key, url in URLS.items():
        flag = getattr(args, f"download_{key}")
        if not flag:
            continue
        path = download_file(key, url, base_dir, chunk_size)
        if args.unzip and key.endswith("_imgs"):
            extract_file(path, base_dir)
            print(f"Extracted {key} into {base_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
