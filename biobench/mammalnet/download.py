# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
A script to download the MammalNet dataset.

Run with:

1. `python biobench/mammalnet/download.py --help` if `biobench/` is in your $PWD.
2. `python -m biobench.mammalnet.download --help` if you have installed `biobench` as a package.
"""

import dataclasses
import os.path
import tarfile

import requests
import tqdm
import tyro

videos_url = (
    "https://mammalnet.s3.amazonaws.com/trimmed_video.tar.gz"
)
labels_url = "https://mammalnet.s3.amazonaws.com/annotation.tar"


@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""

    videos: bool = True
    """Whether to download videos [148GB]."""
    labels: bool = True
    """Whether to download labels."""


def main(args: Args):
    """Download MammalNet."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    videos_tar_path = os.path.join(args.dir, "trimmed_video.tar.gz")
    labels_tar_path = os.path.join(args.dir, "annotation.tar")
    videos_dir_name = "trimmed_video"
    videos_dir_path = os.path.join(args.dir, videos_dir_name)
    labels_dir_name = "annotation"
    labels_dir_path = os.path.join(args.dir, labels_dir_name)

    if args.labels:
        # Download labels
        r = requests.get(labels_url, stream=True)
        r.raise_for_status()

        with open(labels_tar_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        print(f"Downloaded labels: {labels_tar_path}.")

    if args.videos:
        # Download videos.
        r = requests.get(videos_url, stream=True)
        r.raise_for_status()

        n_bytes = int(r.headers["content-length"])

        with open(videos_tar_path, "wb") as fd:
            for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=chunk_size),
                total=n_bytes / chunk_size,
                unit="b",
                unit_scale=1,
                unit_divisor=1024,
                desc="Downloading videos",
            ):
                fd.write(chunk)
        print(f"Downloaded videos: {videos_tar_path}.")

    with tarfile.open(labels_tar_path, "r") as tar:
        tar.extractall(path=args.dir)
    print(f"Extracted labels: {labels_dir_path}.")

    n_videos = 0
    all_video_files = []
    for csv_file in ["train.csv", "test.csv"]:
        with open(os.path.join(labels_dir_path, "composition", csv_file)) as fd:
            video_files = fd.readlines()
            video_files = [video_file.split(" ")[0] for video_file in video_files]
            video_files = [video_file[:13] + video_file[14:] for video_file in video_files]
            all_video_files += video_files
            n_videos += len(video_files)

    with tarfile.open(videos_tar_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting videos", total=n_videos):
            if member.name in all_video_files or member.name == "trimmed_video":
                tar.extract(member, path=args.dir)
    print(f"Extracted videos: {videos_dir_path}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
