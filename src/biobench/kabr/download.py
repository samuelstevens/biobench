# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
Download the Kenyan Animal Behavior Recognition (KABR) dataset.

Examples
--------
# bare-bones (all animals, default path):
python -m biobench.kabr.download

# custom output directory, keep zip archives:
python -m biobench.kabr.download --dir /scratch/KABR --keep-archives
"""

import collections.abc
import dataclasses
import glob
import hashlib
import os.path
import pathlib
import zipfile

import beartype
import requests
import tqdm
import tyro

# --------- #
# Constants #
# --------- #

BASE_URL = "https://huggingface.co/datasets/imageomics/KABR/resolve/main/KABR"
DATASET_PREFIX = "dataset/image/"

ANIMAL_PART_RANGE: dict[str, tuple[str, str]] = {
    "giraffes": ("aa", "ad"),
    "zebras_grevys": ("aa", "am"),
    "zebras_plains": ("aa", "al"),
}

STATIC_FILES: list[str] = [
    "README.txt",
    "annotation/classes.json",
    "annotation/distribution.xlsx",
    "annotation/train.csv",
    "annotation/val.csv",
    "configs/I3D.yaml",
    "configs/SLOWFAST.yaml",
    "configs/X3D.yaml",
    "dataset/image2video.py",
    "dataset/image2visual.py",
]

# ------- #
# Helpers #
# ------- #


@beartype.beartype
def generate_part_files(animal: str, start: str, end: str) -> list[str]:
    """Generate `dataset/image/{animal}_part_??` blocks inclusive of start/end."""
    start_a, start_b = map(ord, start)
    end_a, end_b = map(ord, end)
    return [
        f"{DATASET_PREFIX}{animal}_part_{chr(a)}{chr(b)}"
        for a in range(start_a, end_a + 1)
        for b in range(start_b, end_b + 1)
    ]


@beartype.beartype
def all_files_for_animals(animals: collections.abc.Iterable[str]) -> list[str]:
    files: list[str] = STATIC_FILES.copy()
    for animal in animals:
        files.append(f"{DATASET_PREFIX}{animal}_md5.txt")
        start, end = ANIMAL_PART_RANGE[animal]
        files.extend(generate_part_files(animal, start, end))
    return files


@beartype.beartype
def stream_download(url: str, dst: pathlib.Path, chunk_bytes: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dst, "wb") as fd,
            tqdm.tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dst.name,
                leave=False,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=chunk_bytes):
                fd.write(chunk)
                pbar.update(len(chunk))


@beartype.beartype
def md5_file(path: pathlib.Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    dir: pathlib.Path = pathlib.Path("KABR_files")
    """Where to place downloaded data."""
    animals: tuple[str, ...] = tuple(ANIMAL_PART_RANGE.keys())
    """Subset of animals to download."""
    chunk_size_kb: int = 1024
    """Stream chunk size in KB."""
    keep_archives: bool = False
    """Keep concatenated *.zip files & md5 after extraction."""
    skip_existing: bool = True
    """Skip download if file already present on disk."""


@beartype.beartype
def main(args: Args) -> None:
    files = all_files_for_animals(args.animals)
    chunk = args.chunk_size_kb * 1024

    print("Downloading KABR ...")
    for rel in tqdm.tqdm(files, unit="file"):
        dst = args.dir / rel
        if args.skip_existing and dst.exists():
            continue
        url = f"{BASE_URL}/{rel}"
        stream_download(url, dst, chunk)

    print("Concatenating split archives ...")
    for animal in args.animals:
        out_zip = args.dir / f"{DATASET_PREFIX}{animal}.zip"
        parts = sorted(
            glob.glob(str(args.dir / f"{DATASET_PREFIX}{animal}_part_*")),
            key=lambda p: pathlib.Path(p).name,
        )
        if out_zip.exists() or not parts:
            continue
        total_bytes = sum(os.path.getsize(p) for p in parts)
        with (
            open(out_zip, "wb") as dst,
            tqdm.tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Concat {animal}",
                leave=False,
            ) as bar,
        ):
            for part in parts:
                with open(part, "rb") as src:
                    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                        dst.write(chunk)
                        bar.update(len(chunk))
                pathlib.Path(part).unlink()

    print("Validating & extracting ...")
    for animal in tqdm.tqdm(args.animals, unit="animal"):
        md5_txt = args.dir / f"{DATASET_PREFIX}{animal}_md5.txt"
        zip_path = args.dir / f"{DATASET_PREFIX}{animal}.zip"
        if not md5_txt.exists() or not zip_path.exists():
            print(f"Skipping {animal} (missing files).")
            continue

        expected = md5_txt.read_text().strip().split()[0]
        got = md5_file(zip_path)
        if got != expected:
            raise RuntimeError(
                f"MD5 mismatch for {zip_path.name}: {got} (expected {expected})"
            )

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)

        if not args.keep_archives:
            zip_path.unlink(missing_ok=True)
            md5_txt.unlink(missing_ok=True)

    print(f"Done. Data at: {args.dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
