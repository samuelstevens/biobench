# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
Downloads the Begula whale dataset from lila.science.
"""

import dataclasses
import os
import tarfile

import requests
import tqdm
import tyro

train_url = "http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/wild-me/beluga.coco.tar.gz"


@dataclasses.dataclass(frozen=True)
class Args:
    """Configuration."""

    dir: str = "."
    """where to save data."""
    chunk_size_kb: int = 1
    """how many KB to download at a time before writing to file."""
    download: bool = True
    """whether to download images."""
    expand: bool = True
    """whether to expand tarfiles into a folder."""


def main(args: Args):
    """Download and unzip the data."""
    os.makedirs(args.dir, exist_ok=True)

    chunk_size = int(args.chunk_size_kb * 1024)

    images_tar_path = os.path.join(args.dir, "beluga.coco.tar.gz")

    if args.download:
        # Download images.
        r = requests.get(train_url, stream=True)
        r.raise_for_status()
        t = tqdm.tqdm(
            total=int(r.headers["content-length"]),
            unit="B",
            unit_scale=1,
            unit_divisor=1024,
            desc="Downloading images",
        )
        with open(images_tar_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
                t.update(len(chunk))
        t.close()

        print(f"Downloaded images: {images_tar_path}.")

    if args.expand:
        with tarfile.open(images_tar_path, "r") as tar:
            for member in tqdm.tqdm(
                tar, desc="Extracting images", total=len(tar.getnames())
            ):
                tar.extract(member, path=args.dir, filter="data")

        print(f"Extracted images: {args.dir}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
