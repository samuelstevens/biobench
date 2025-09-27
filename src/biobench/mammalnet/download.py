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
import math
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
    download_videos: bool = False
    """Whether to download the video archive."""
    download_annotations: bool = False
    """Whether to download the annotation archive."""
    trim_videos: bool = False
    """Whether to create trimmed video clips based on annotations."""
    check_stats: bool = False
    """Whether to check video length statistics for both full and trimmed videos."""
    sample_frames: bool = False
    """Whether to extract stills from trimmed clips."""
    n_frames: int = 32
    """How many frames per clip."""
    n_workers: int = 16
    """Number of parallel `ffmpeg`s to spawn."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Annotation:
    label: str
    """The class label for this annotation segment."""
    start_s: float
    """Start time in seconds."""
    end_s: float
    """End time in seconds."""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Detection:
    vid_id: str
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
    def from_json(cls, vid_id, dct):
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
            vid_id=vid_id,
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
def _duration(path: pathlib.Path) -> float:
    out = ffmpeg.probe(str(path))
    return float(out["format"]["duration"])


@beartype.beartype
def _n_frames(path: pathlib.Path, *, tol: float = 0.01) -> int:
    """
    Robust frame-count extractor.

    Priority:
        1.   stream.nb_frames                      (exact when present)
        2-4. derived counts that should agree within *tol*
             * duration_ts / time_base
             * duration * fps
             * coded_number_of_frames (if present)
    Returns the majority (or first) value that satisfies pair-wise agreement.
    Raises if no two estimates agree.

    tol: relative tolerance, e.g. 0.01 -> 1 % mismatch allowed.
    """
    meta = ffmpeg.probe(str(path))["streams"][0]

    def as_int(x: str | None) -> int | None:
        return int(x) if x and x.isdigit() else None

    candidates = {}

    # 1) nb_frames (often absent on H.264)
    if n := as_int(meta.get("nb_frames")):
        candidates["nb_frames"] = n

    # 2) duration_ts / time_base
    if meta.get("duration_ts") and meta.get("time_base"):
        num_ts = int(meta["duration_ts"])
        tb_num, tb_den = map(int, meta["time_base"].split("/"))
        if tb_num:
            candidates["duration_ts"] = round(num_ts * tb_den / tb_num)

    # 3) duration * fps
    fps_num, fps_den = map(int, meta["r_frame_rate"].split("/"))
    fps = fps_num / fps_den if fps_den else 0
    if meta.get("duration") and fps:
        candidates["duration*fps"] = round(float(meta["duration"]) * fps)

    # 4) coded_number_of_frames (for some codecs)
    if n := as_int(meta.get("coded_number_of_frames")):
        candidates["coded_frames"] = n

    # choose majority-agreeing integer
    votes = {}
    for _, v in candidates.items():
        votes[v] = votes.get(v, 0) + 1

    # allow near-duplicates
    for i_name, i_val in candidates.items():
        if any(
            math.isclose(i_val, j_val, rel_tol=tol)
            for j_val in votes.keys()
            if j_val != i_val
        ):
            votes[i_val] += 1

    best, count = max(votes.items(), key=lambda kv: kv[1])
    if count < 2 and len(candidates) > 1:  # no agreement
        raise RuntimeError(
            f"Inconsistent frame counts for {path.name}: "
            + ", ".join(f"{k}={v}" for k, v in candidates.items())
        )
    return best


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
        futs = [pool.submit(_duration, vid) for vid in vids]
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
        futs = [pool.submit(_duration, vid) for vid in vids]
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
def _sample(src: pathlib.Path, dst_dir: pathlib.Path, *, n_frames: int):
    if dst_dir.exists() and any(dst_dir.iterdir()):
        return  # already done

    step = max(_n_frames(src) // n_frames, 1)  # stride in frame space

    dst_dir.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(src))
        .filter("select", f"not(mod(n,{step}))")
        .filter("setpts", "N/FRAME_RATE/TB")
        .output(
            str(dst_dir / "frame_%02d.jpg"),
            vframes=n_frames,
            vsync="vfr",
            qscale=2,
            loglevel="error",
            **{"an": None},
        )
        .overwrite_output()
        .run()
    )


@beartype.beartype
def _sample_frames(base: pathlib.Path, *, n_workers: int, n_frames: int):
    trimmed = base / "trimmed_videos"
    out = base / "frames"
    vids = [p for p in trimmed.iterdir() if p.suffix.lower() == ".mp4"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_sample, p, out / p.stem, n_frames=n_frames) for p in vids]
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

    if args.sample_frames:
        _sample_frames(base, n_workers=args.n_workers, n_frames=args.n_frames)


if __name__ == "__main__":
    main(tyro.cli(Args))
