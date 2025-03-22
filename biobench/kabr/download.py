# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
import concurrent.futures
import glob
import hashlib
import logging
import os
import zipfile

import beartype
import requests
import tqdm
import tyro

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("biobench")

base_url = "https://huggingface.co/datasets/imageomics/KABR/resolve/main/KABR"


@beartype.beartype
def generate_part_files(animal: str, start, end) -> list[str]:
    start_a, start_b = ord(start[0]), ord(start[1])
    end_a, end_b = ord(end[0]), ord(end[1])
    return [
        f"{animal}_part_{chr(a)}{chr(b)}"
        for a in range(start_a, end_a + 1)
        for b in range(start_b, end_b + 1)
    ]


@beartype.beartype
def concatenate_files(out: str, animal: str):
    logger.info("Concatenating files for '%s'.", animal)

    part_files = sorted(glob.glob(os.path.join(out, f"{animal}_part_*")))
    assert part_files
    breakpoint()
    with open(os.path.join(out, f"{animal}.zip"), "wb") as f_out:
        for fpath in part_files:
            # Simplify this copy-paste code that concats multiple binary files together. AI!
            with open(fpath, "rb") as f_in:
                # Read and write in chunks
                CHUNK_SIZE = 8 * 1024 * 1024  # 8MB
                for chunk in iter(lambda: f_in.read(CHUNK_SIZE), b""):
                    f_out.write(chunk)
            # Delete part files as they are concatenated
            os.remove(fpath)
    logger.info("Archive for '%s' concatenated.", animal)


def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        CHUNK_SIZE = 8 * 1024 * 1024  # 8MB
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_and_extract(animal):
    print(f"Confirming data integrity for {animal}.zip ...")
    zip_md5 = compute_md5(f"{save_dir}/{dataset_prefix}{animal}.zip")

    with open(f"{save_dir}/{dataset_prefix}{animal}_md5.txt", "r") as file:
        expected_md5 = file.read().strip().split()[0]

    if zip_md5 == expected_md5:
        print(f"MD5 sum for {animal}.zip is correct.")

        print(f"Extracting {animal}.zip ...")
        with zipfile.ZipFile(
            f"{save_dir}/{dataset_prefix}{animal}.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(f"{save_dir}/{dataset_prefix}")
        print(f"{animal}.zip extracted.")
        print(f"Cleaning up for {animal} ...")
        os.remove(f"{save_dir}/{dataset_prefix}{animal}.zip")
        os.remove(f"{save_dir}/{dataset_prefix}{animal}_md5.txt")
    else:
        print(
            f"MD5 sum for {animal}.zip is incorrect. Expected: {expected_md5}, but got: {zip_md5}."
        )
        print(
            "There may be data corruption. Please try to download and reconstruct the data again or reach out to the corresponding authors for assistance."
        )


@beartype.beartype
def main(out: str):
    animals = ["giraffes", "zebras_grevys", "zebras_plains"]

    animal_parts_range = {
        "giraffes": ("aa", "ad"),
        "zebras_grevys": ("aa", "am"),
        "zebras_plains": ("aa", "al"),
    }

    # Define the static files that are not dependent on the animals list
    static_files = [
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

    # Generate the part files for each animal
    part_files = [
        part
        for animal, (start, end) in animal_parts_range.items()
        for part in generate_part_files(animal, start, end)
    ]

    archive_md5_files = [f"{animal}_md5.txt" for animal in animals]

    files = static_files + archive_md5_files + part_files

    # Loop through each relative file path

    logger.info("Downloading the Kenyan Animal Behavior Recognition (KABR) dataset.")

    for i, fname in enumerate(tqdm.tqdm(files)):
        # Construct the full URL
        fpath = os.path.join(out, fname)

        if os.path.exists(fpath):
            logger.debug("File '%s' exists. Skipping download.", fpath)
            continue

        url = f"{base_url}/{fname}"

        # Create the necessary directories based on the file path
        os.makedirs(os.path.join(out, os.path.dirname(fpath)), exist_ok=True)

        # Download the file and save it with the preserved file path
        response = requests.get(url)
        with open(fpath, "wb") as file:
            file.write(response.content)

    with concurrent.futures.ThreadPoolExecutor() as pool:
        list(pool.map(lambda animal: concatenate_files(out, animal), animals))

    with concurrent.futures.ThreadPoolExecutor() as pool:
        list(pool.map(verify_and_extract, animals))


if __name__ == "__main__":
    tyro.cli(main)
