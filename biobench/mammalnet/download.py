# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "ffmpeg-python",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
Download the MammalNet benchmark and its annotations.
"""

import concurrent.futures
import dataclasses
import json
import logging
import pathlib
import statistics
import tarfile

import beartype
import ffmpeg
import requests
import tqdm
import tyro

VIDEOS_URL = "https://mammalnet.s3.amazonaws.com/full_video.tar.gz"
ANNOTATIONS_URL = "https://mammalnet.s3.amazonaws.com/annotation.tar"

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("mammalnet.download")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    dir: str = "."
    """Where to save the downloaded archives and (optionally) extract them."""
    chunk_size_kb: int = 1024
    """Download chunk size (KB). 1024 KB ~ 1 MB."""
    download_videos: bool = True
    """Whether to download the video archive."""
    download_annotations: bool = True
    """Whether to download the annotation archive."""
    trim_videos: bool = True
    """Whether to create trimmed video clips based on annotations."""
    check_stats: bool = True
    n_workers: int = 16
    """Number of parallel `ffmpeg`s to spawn."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Annotation:
    label: str
    """The class label for this annotation segment."""
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Detection:
    id: str
    """Unique identifier for the video clip."""
    taxonomy: list[dict[str, str]]
    """Taxonomic classification information for the detected animal."""
    annotations: list[Annotation]
    """List of time segments with behavior annotations."""
    duration_s: int
    """Total duration of the video in seconds."""
    resolution: tuple[int, int]
    """Video resolution in pixels"""
    fps: int
    """Frames per second of the video."""
    subset: str
    """Dataset split this video belongs to (e.g., 'train', 'val', 'test')."""
    url: str
    """Original source URL for the video."""

    @classmethod
    def from_json(cls, id, dct):
        annotations = [
            Annotation(
                label=ann["label"],
                start_s=float(ann["segment"][0]),
                end_s=float(ann["segment"][1]),
            )
            for ann in dct.pop("annotations")
        ]
        taxonomy = dct.pop("taxnomy")
        duration_s = dct.pop("duration")
        resolution = tuple(int(x) for x in dct.pop("resolution").split("x"))
        return cls(
            id=id,
            taxonomy=taxonomy,
            annotations=annotations,
            duration_s=duration_s,
            resolution=resolution,
            **dct,
        )


@beartype.beartype
def _download(url: str, dest: pathlib.Path, chunk_bytes: int) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest.name} already present, skipping download.")
        return dest

    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    with (
        dest.open("wb") as f,
        tqdm.tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
        ) as bar,
    ):
        for chunk in r.iter_content(chunk_size=chunk_bytes):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


@beartype.beartype
def _extract(archive: pathlib.Path, out_dir: pathlib.Path) -> None:
    with tarfile.open(archive, "r:*") as tar:
        for member in tqdm.tqdm(tar, desc=f"Extracting {archive.name}"):
            tar.extract(member, path=out_dir)


@beartype.beartype
def _probe(path: pathlib.Path) -> float:
    out = ffmpeg.probe(str(path))
    return float(out["format"]["duration"])


@beartype.beartype
def _load_detections(base: pathlib.Path) -> list[Detection]:
    with open(base / "annotation" / "detection_annotations.json") as fd:
        detections = [
            Detection.from_json(key, value) for key, value in json.load(fd).items()
        ]

    return detections


@beartype.beartype
def _stats(base: pathlib.Path, n_workers: int = 8):
    detections = _load_detections(base)

    ###############
    # Full videos #
    ###############

    from_json = [det.duration_s for det in detections]
    mean_s_from_json = statistics.mean(from_json)

    vids = list(
        p for p in (base / "full_videos").iterdir() if p.suffix.lower() == ".mp4"
    )
    durations = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_probe, vid) for vid in vids]
        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="full video durations",
        ):
            durations.append(fut.result())
    mean_s_from_disk = statistics.mean(durations)

    print("From paper:")
    print(f"  Mean (s) : {106:6.1f}")
    print("From detection_annotations.json:")
    print(f"  Mean (s) : {mean_s_from_json:6.1f}")
    print(f"From {base / 'full_videos'}:")
    print(f"  Mean (s) : {mean_s_from_disk:6.1f}")

    ##################
    # Trimmed videos #
    ##################

    # Calculate expected durations from annotations
    from_json = [ann.duration_s for det in detections for ann in det.annotations]
    mean_s_from_json = statistics.mean(from_json)

    vids = list(
        p for p in (base / "trimmed_videos").iterdir() if p.suffix.lower() == ".mp4"
    )
    durations = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_probe, vid) for vid in vids]
        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="trimmed video durations",
        ):
            if err := fut.exception():
                logger.warning("Exception: %s", err)
                continue
            durations.append(fut.result())
    mean_s_from_disk = statistics.mean(durations)

    print("From paper:")
    print(f"  Mean (s) : {77:6.1f}")
    print("From detection_annotations.json:")
    print(f"  Mean (s) : {mean_s_from_json:6.1f}")
    print(f"From {base / 'trimmed_videos'}:")
    print(f"  Mean (s) : {mean_s_from_disk:6.1f}")


@beartype.beartype
def _trim(src: pathlib.Path, dst: pathlib.Path, start_s: float, end_s: float):
    """Copy-trim [start_s, end_s] from *src* into *dst* without re-encoding."""
    if start_s >= end_s:
        raise ValueError("start_s must be < end_s")

    duration = end_s - start_s
    (
        ffmpeg
        # fast seek to ~start (input-side -ss is key-frame aligned, faster)
        .input(str(src), ss=start_s)
        # output-side -t gives exact length; libx264/AAC re-encodes safely
        .output(
            str(dst),
            t=duration,
            vcodec="libx264",
            crf=23,
            preset="veryfast",
            movflags="+faststart",  # web-friendly moov placement
            loglevel="error",  # silence ffmpeg spam unless errors
            **{"an": None},  # <- -an  (strip audio)
        )
        .overwrite_output()
        .run()
    )


@beartype.beartype
def _trim_all(base: pathlib.Path, n_workers: int):
    (base / "trimmed_videos").mkdir(exist_ok=True)
    jobs = []
    with open(base / "annotation" / "detection_annotations.json") as fd:
        for key, value in json.load(fd).items():
            det = Detection.from_json(key, value)

            src = base / "full_videos" / f"{det.id}.mp4"
            if len(det.annotations) > 1:
                for k, ann in enumerate(det.annotations):
                    dst = base / "trimmed_videos" / f"{det.id}_{k + 1}.mp4"
                    jobs.append((src, dst, ann.start_s, ann.end_s))
            else:
                ann = det.annotations[0]
                dst = base / "trimmed_videos" / f"{det.id}.mp4"
                jobs.append((src, dst, ann.start_s, ann.end_s))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_trim, *job) for job in jobs]
        for fut in tqdm.tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
            fut.result()  # re-raise on failure


@beartype.beartype
def main(args: Args) -> None:
    base = pathlib.Path(args.dir).expanduser().resolve()
    chunk = args.chunk_size_kb * 1024

    if args.download_videos:
        target = base / pathlib.Path(VIDEOS_URL).name
        archive = _download(VIDEOS_URL, target, chunk)

        _extract(archive, base)
        print(f"Extracted videos into {base}")

    if args.download_annotations:
        target = base / pathlib.Path(ANNOTATIONS_URL).name
        archive = _download(ANNOTATIONS_URL, target, chunk)

        _extract(archive, base)
        print(f"Extracted annotations into {base}")

    if args.trim_videos:
        _trim_all(base, args.n_workers)

    if args.check_stats:
        _stats(base, args.n_workers)


if __name__ == "__main__":
    main(tyro.cli(Args))
