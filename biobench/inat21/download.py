# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beartype",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
A script to download the iNat21 (mini) dataset.

Run with:

1. `python biobench/inat21/download.py --help` if `biobench/` is in your $PWD.
2. `python -m biobench.inat21.download --help` if you have installed `biobench` as a package.
"""

import dataclasses
import os.path
import tarfile

import beartype
import requests
import tqdm
import tyro

val_images_url = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"
train_mini_images_url = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"
)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""

    val: bool = True
    """Whether to download validation images [8.4GB]."""
    train: bool = True
    """Whether to download (mini) train images [42GB]."""


@beartype.beartype
def download_tar(url: str, tar_path: str, chunk_size: int):
    r = requests.get(url, stream=True)
    r.raise_for_status()

    n_bytes = int(r.headers["content-length"])

    with open(tar_path, "wb") as fd:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=n_bytes / chunk_size,
            unit="b",
            unit_scale=1,
            unit_divisor=1024,
            desc="Downloading",
        ):
            fd.write(chunk)


def extract_tar(tar_path: str, n_images: int, dir: str):
    with tarfile.open(tar_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting images", total=n_images + 10_000):
            tar.extract(member, path=dir, filter="data")


@beartype.beartype
def main(args: Args):
    """Download NeWT."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    train_tar_path = os.path.join(args.dir, "train_mini.tar.gz")
    val_tar_path = os.path.join(args.dir, "val.tar.gz")

    if args.val:
        download_tar(val_images_url, val_tar_path, chunk_size)
        print(f"Downloaded validation images: {val_tar_path}.")

    extract_tar(val_tar_path, 100_000, args.dir)
    print("Extracted validation images.")

    if args.train:
        download_tar(train_mini_images_url, train_tar_path, chunk_size)
        print(f"Downloaded train (mini) images: {train_tar_path}.")

    extract_tar(train_tar_path, 500_000, args.dir)
    print("Extracted training images.")


if __name__ == "__main__":
    main(tyro.cli(Args))
