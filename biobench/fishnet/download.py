# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "gdown",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
A script to download the FishNet dataset

Run with:

1. `python biobench/fishnet/download.py --help` if `biobench/` is in your $PWD.
2. `python -m biobench.fishnet.download --help` if you have installed `biobench` as a package.
"""

import dataclasses
import os.path
import zipfile

import requests
import gdown
import tqdm
import tyro

dataset_url = "https://drive.google.com/uc?id=1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3"

labels_urls = [
    "https://raw.githubusercontent.com/faixan-khan/FishNet/refs/heads/main/anns/train_full_meta_new.csv",
    "https://raw.githubusercontent.com/faixan-khan/FishNet/refs/heads/main/anns/train.csv",
    "https://raw.githubusercontent.com/faixan-khan/FishNet/refs/heads/main/anns/test.csv",
    "https://raw.githubusercontent.com/faixan-khan/FishNet/refs/heads/main/anns/spec_gen_map.csv"
]


@dataclasses.dataclass()
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""

    images: bool = True
    """Whether to download the zip file [5.4GB]."""
    labels: bool = True
    """Whether to download the labels."""
    extract: bool = True
    """Whether to extract the zip file."""


def main(args: Args):
    """Download FishNet."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    output_name = "fishnet.zip"
    zipfile_path = os.path.join(args.dir, output_name)

    # Download the zip file.
    if args.images:
        gdown.download(dataset_url, zipfile_path, quiet=False)
        print(f"Downloaded zip file: {zipfile_path}.")
    
    if args.labels:
        for labels_url in labels_urls:
            r = requests.get(labels_url, stream=True)
            r.raise_for_status()

            labels_path = os.path.join(args.dir, labels_url.split("/")[-1])
            with open(labels_path, "wb") as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
            print(f"Downloaded labels: {labels_path}.")

    # Extract the zip file.
    if args.extract:
        with zipfile.ZipFile(zipfile_path, 'r') as zip:
            for member in tqdm.tqdm(zip.infolist(), desc="Extracting images"):
                zip.extract(member, args.dir)
        print(f"Extracted images: {args.dir}/Image_Library.")


if __name__ == "__main__":
    main(tyro.cli(Args))
