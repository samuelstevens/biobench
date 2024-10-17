# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
A script to download the SYKE-plankton_IFCB_2022 dataset.

Run with:

1. `python biobench/plankton/download.py --help` if `biobench/` is in your $PWD.
2. `python -m biobench.plankton.download --help` if you have installed `biobench` as a package.
"""

import dataclasses
import os
import shutil
import zipfile

import requests
import tqdm
import tyro

train_url = "https://b2share.eudat.eu/api/files/63a79aff-4194-48c8-8055-0a73ecfcf183/phytoplankton_labeled.zip"
val_url = "https://b2share.eudat.eu/api/files/4a62bb1b-9bd0-4005-9217-7472ee6ed92c/phytoplankton_Ut%C3%B6_2021_labeled.zip"


@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""


def main(args: Args):
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    train_zip = os.path.join(args.dir, "train.zip")
    val_zip = os.path.join(args.dir, "val.zip")

    for filepath, url in [(train_zip, train_url), (val_zip, val_url)]:
        r = requests.get(url, stream=True)
        r.raise_for_status()

        n_bytes = int(r.headers["content-length"])

        with open(filepath, "wb") as fd:
            # Need to specify a manual progress bar in order to get units and such working.
            t = tqdm.tqdm(
                total=n_bytes,
                unit="B",
                unit_scale=1,
                unit_divisor=1024,
                desc="Downloading images",
            )
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
                t.update(len(chunk))
            t.close()

        with zipfile.ZipFile(filepath, "r") as zip:
            for member in tqdm.tqdm(
                zip.infolist(), unit="img", desc="Extracting images"
            ):
                zip.extract(member, args.dir)

    # Move images to particular split-named folders.
    val_folder = "phytoplankton_Utö_2021_labeled"
    move(os.path.join(args.dir, val_folder), os.path.join(args.dir, "val"))
    train_folder = "labeled_20201020"
    move(os.path.join(args.dir, train_folder), os.path.join(args.dir, "train"))

    print(f"Downloaded, extracted and organized images in {args.dir}.")


def move(src: str, dst: str):
    """
    Moves _src_ to _dst_. If _dst_ exists, it will be overwritten.
    """
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.rename(src, dst)


if __name__ == "__main__":
    main(tyro.cli(Args))
