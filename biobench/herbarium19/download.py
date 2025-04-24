# biobench/herbarium19/download.py
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
Download the Herbarium 2019 FGVC6 dataset directly from storage.googleapis.com.

Splits:
 - train (34,225 images; 38 GB)
 - validation (2,679 images; 3 GB)
 - test (9,565 images; 11 GB)

Verifies MD5 checksums before extraction.
"""

import dataclasses
import hashlib
import os
import tarfile

import requests
import tqdm
import tyro

URLS = {
    "train": {
        "url": "https://storage.googleapis.com/nybg/herbarium-2019-fgvc6/train.tar.gz",
        "md5": "53c6b9ee2f831f5101dbe00958091dc8",
    },
    "validation": {
        "url": "https://storage.googleapis.com/nybg/herbarium-2019-fgvc6/validation.tar.gz",
        "md5": "2f854d580949e54f114993a74adc3d4b",
    },
    "test": {
        "url": "https://storage.googleapis.com/nybg/herbarium-2019-fgvc6/test.tar.gz",
        "md5": "297648fb76eed1b1c6f0ca1fd8188de0",
    },
}


@dataclasses.dataclass(frozen=True)
class Args:
    dir: str = "."
    """Where to save the downloaded archives and extract them."""
    chunk_size_kb: int = 1024
    """Download chunk size in KB."""
    download_train: bool = True
    download_validation: bool = True
    download_test: bool = True
    unzip: bool = True


def md5_of_file(path: str, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        desc = f"MD5 for {os.path.basename(path)}"
        with tqdm.tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as bar:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
                bar.update(len(chunk))
    return h.hexdigest()


def download_split(name: str, dest_dir: str, chunk_size: int) -> str:
    info = URLS[name]
    url, expected_md5 = info["url"], info["md5"]
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, f"{name}.tar.gz")

    # Skip download if present and checksum matches
    if os.path.exists(archive_path):
        if md5_of_file(archive_path) == expected_md5:
            print(f"{name}.tar.gz already exists and MD5 matches; skipping download")
            return archive_path
        else:
            print(f"MD5 mismatch for existing {name}.tar.gz; re-downloading")

    # Stream-download
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with (
        open(archive_path, "wb") as f,
        tqdm.tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download {name}"
        ) as bar,
    ):
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    # verify
    actual_md5 = md5_of_file(archive_path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {name}.tar.gz: expected {expected_md5}, got {actual_md5}"
        )

    return archive_path


def extract_split(archive_path: str, dest_dir: str):
    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tqdm.tqdm(tar, desc=f"Extracting {os.path.basename(archive_path)}"):
            tar.extract(m, path=dest_dir)


def main(args: Args):
    splits = {
        "train": args.download_train,
        "validation": args.download_validation,
        "test": args.download_test,
    }
    for name, do_dl in splits.items():
        if not do_dl:
            continue
        archive = download_split(name, args.dir, args.chunk_size_kb * 1024)
        if args.unzip:
            extract_split(archive, args.dir)
            print(f"Extracted `{name}` into {args.dir}/{name}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
