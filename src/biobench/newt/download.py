# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
A script to download the NeWT dataset.

Run with:

1. `python biobench/newt/download.py --help` if `biobench/` is in your $PWD.
2. `python -m biobench.newt.download --help` if you have installed `biobench` as a package.
"""

import dataclasses
import os.path
import tarfile

import requests
import tqdm
import tyro

images_url = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_images.tar.gz"
)
labels_url = "https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_labels.csv.tar.gz"


@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""

    images: bool = True
    """Whether to download images [4.1GB]."""
    labels: bool = True
    """Whether to download labels."""


def main(args: Args):
    """Download NeWT."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    labels_tar_path = os.path.join(args.dir, "labels.tar")
    images_tar_path = os.path.join(args.dir, "images.tar")
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(args.dir, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(args.dir, images_dir_name)

    if args.labels:
        # Download labels
        r = requests.get(labels_url, stream=True)
        r.raise_for_status()

        with open(labels_tar_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        print(f"Downloaded labels: {labels_tar_path}.")

    if args.images:
        # Download images.
        r = requests.get(images_url, stream=True)
        r.raise_for_status()

        n_bytes = int(r.headers["content-length"])

        with open(images_tar_path, "wb") as fd:
            for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=chunk_size),
                total=n_bytes / chunk_size,
                unit="b",
                unit_scale=1,
                unit_divisor=1024,
                desc="Downloading images",
            ):
                fd.write(chunk)
        print(f"Downloaded images: {images_tar_path}.")

    with tarfile.open(labels_tar_path, "r") as tar:
        tar.extract(labels_csv_name, path=args.dir, filter="data")
    print(f"Extracted labels: {labels_csv_path}.")

    with open(labels_csv_path) as fd:
        n_images = len(fd.read().split("\n")) - 1

    with tarfile.open(images_tar_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting images", total=n_images):
            tar.extract(member, path=args.dir, filter="data")
    print(f"Extracted images: {images_dir_path}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
