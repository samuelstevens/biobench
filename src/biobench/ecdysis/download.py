# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "beautifulsoup4",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
Script to download Ecdysis Foundation's public dataset.

Run with:

1. `python biobench/ecdysis/download.py --help` if `biobench/` is in your $PWD.
2. `python -m biobench.ecdysis.download --help` if you have installed `biobench` as a package.
"""

import concurrent.futures
import csv
import dataclasses
import logging
import pathlib
import zipfile

import beartype
import bs4
import requests
import tqdm
import tyro

download_page_url = "https://bugbox.ecdysis.bio/samples/collection-download/1"
# zipped csv file with columns
# 1   id
# 2   specimen_id
# 3   image_thumbnail_large
# 4   archival_identifier
# 5   archival_stored
# 6   visit_date
# 7   country
# 8   state_region
# 9   county_region
# 10  us_state_county_fips
# 11  gbif_class
# 12  gbif_order
# 13  gbif_family
# 14  gbif_genus
# 15  gbif_species
# 16  reviewed
# 17  public_url


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""

    images: bool = True
    """Whether to download images."""
    metadata: bool = True
    """Whether to download metadata."""

    n_workers: int = 32
    """Number of parallel downloads for images."""


@beartype.beartype
def get_csv_url() -> str:
    """Fetch the download page and extract the dynamically generated CSV URL."""
    response = requests.get(download_page_url)
    response.raise_for_status()

    soup = bs4.BeautifulSoup(response.text, "html.parser")

    csv_link = soup.find("a", href=lambda h: h and ".csv.zip" in h)
    if csv_link is None:
        raise ValueError(f"Could not find CSV download link on {download_page_url}")

    csv_url = csv_link["href"]
    return csv_url


@beartype.beartype
def download_image(url: str, dst_fpath: pathlib.Path) -> pathlib.Path | None:
    """Download a single image from URL to destination path."""
    if dst_fpath.exists():
        return None

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        dst_fpath.parent.mkdir(parents=True, exist_ok=True)

        with open(dst_fpath, "wb") as fd:
            fd.write(response.content)

        return dst_fpath
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to download {url}: {e}")
        return None


@beartype.beartype
def download_all_images(
    csv_fpath: pathlib.Path, images_dpath: pathlib.Path, n_workers: int
) -> None:
    """Download all images from CSV file using parallel workers."""
    logger = logging.getLogger(__name__)

    with open(csv_fpath, "r") as fd:
        reader = csv.DictReader(fd)
        rows = list(reader)

    logger.info(f"Found {len(rows)} images to download")

    images_dpath.mkdir(parents=True, exist_ok=True)

    jobs = []
    for row in rows:
        url = row["public_url"]
        filename = pathlib.Path(url).name
        dst_fpath = images_dpath / filename
        jobs.append((url, dst_fpath))

    downloaded = 0
    skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(download_image, *job) for job in jobs]
        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="Downloading images",
        ):
            result = fut.result()
            if result is None:
                skipped += 1
            else:
                downloaded += 1

    logger.info(f"Downloaded {downloaded} images, skipped {skipped} existing")


def main(args: Args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dpath = pathlib.Path(args.dir)
    output_dpath.mkdir(parents=True, exist_ok=True)

    csv_fpath = output_dpath / "metadata.csv"
    chunk_size = args.chunk_size_kb * 1024

    if args.metadata:
        logger.info("Fetching CSV download URL...")
        csv_url = get_csv_url()
        logger.info(f"Found CSV URL: {csv_url}")

        zip_fpath = output_dpath / "metadata.csv.zip"
        logger.info(f"Downloading metadata to {zip_fpath}")

        response = requests.get(csv_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(zip_fpath, "wb") as fd:
            with tqdm.tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fd.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Extracting {zip_fpath}")
        with zipfile.ZipFile(zip_fpath, "r") as zip_ref:
            zip_ref.extractall(output_dpath)

        extracted_fpath = output_dpath / "zip"
        if extracted_fpath.exists():
            extracted_fpath.rename(csv_fpath)
            logger.info(f"Renamed extracted file to {csv_fpath}")

        logger.info("Metadata download complete")

    if args.images:
        if not csv_fpath.exists():
            logger.error(
                f"Metadata CSV not found at {csv_fpath}. Download metadata first."
            )
            return

        images_dpath = output_dpath / "images"
        download_all_images(csv_fpath, images_dpath, args.n_workers)


if __name__ == "__main__":
    main(tyro.cli(Args))
