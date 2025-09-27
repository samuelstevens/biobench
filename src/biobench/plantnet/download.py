import dataclasses
import os.path
import zipfile

import requests
import tqdm
import tyro

images_url = "https://zenodo.org/records/5645731/files/plantnet_300K.zip"


@dataclasses.dataclass(frozen=True)
class Args:
    dir: str = "."
    """where to save data."""
    chunk_size_kb: int = 1
    """how many KB to download at a time before writing to file."""
    download: bool = True
    """whether to download images [29.5GB]."""
    unzip: bool = True
    """whether to unzip images."""


def main(args: Args):
    os.makedirs(args.dir, exist_ok=True)

    chunk_size = int(args.chunk_size_kb * 1024)

    images_zip_path = os.path.join(args.dir, "plantnet_300K.zip")

    if args.download:
        # Download images.
        r = requests.get(images_url, stream=True)
        r.raise_for_status()

        t = tqdm.tqdm(
            total=int(r.headers["content-length"]),
            unit="B",
            unit_scale=1,
            unit_divisor=1024,
            desc="Downloading images",
        )
        with open(images_zip_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
                t.update(len(chunk))
        t.close()

        print(f"Downloaded images: {images_zip_path}.")

    if args.unzip:
        # Unzip images.
        zip_file = zipfile.ZipFile(images_zip_path)
        names = zip_file.namelist()
        for filename in tqdm.tqdm(names, desc="Unzipping images."):
            zip_file.extract(filename, path=args.dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
