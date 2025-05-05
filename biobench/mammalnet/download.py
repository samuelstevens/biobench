# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "av",
#     "beartype",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
Download the **MammalNet** trimmed-video benchmark and its annotations.

Sources (as of 2025-05-01)
--------------------------
* Videos (~XX GB):   https://mammalnet.s3.amazonaws.com/trimmed_videos.tar.gz
* Annotations:      https://mammalnet.s3.amazonaws.com/annotation.tar

After downloading you can pass `--unzip true` to extract the archives.
"""

import dataclasses
import json
import pathlib
import tarfile

import av
import beartype
import requests
import tqdm
import tyro

VIDEOS_URL = "https://mammalnet.s3.amazonaws.com/full_video.tar.gz"
ANNOTATIONS_URL = "https://mammalnet.s3.amazonaws.com/annotation.tar"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    dir: str = "."
    """Where to save the downloaded archives and (optionally) extract them."""
    chunk_size_kb: int = 1024
    """Download chunk size (KB). 1024 KB ≈ 1 MB."""
    download_videos: bool = True
    """Whether to download the video archive."""
    download_annotations: bool = True
    """Whether to download the annotation archive."""
    trim_videos: bool = True
    """Whether to create trimmed video clips based on annotations."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Annotation:
    # Document this class with docstrings for each value. AI!
    label: str
    segment: tuple[float, float]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Detection:
    # Document this class with docstrings for each value. AI!
    id: str
    taxonomy: list[dict[str, str]]
    annotations: list[Annotation]
    duration: int
    resolution: str
    fps: int
    subset: str
    url: str


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
def _trim(detection: Detection, src: pathlib.Path, dst: pathlib.Path):
    pass


@beartype.beartype
def _trim(src: pathlib.Path, dst: pathlib.Path, start_s: float, end_s: float):
    """Copy-trim [start_s, end_s] from *src* into *dst* without re-encoding."""
    # open once for reading
    with av.open(str(src)) as in_container:
        video_stream = in_container.streams.video[0]
        time_base = video_stream.time_base  # seconds / pts

        # prepare writer
        with av.open(str(dst), "w") as out_container:
            out_stream = out_container.add_stream_from_template(video_stream)

            start_ts = int(start_s / time_base)
            end_ts = int(end_s / time_base)

            in_container.seek(
                start_ts, any_frame=False, backward=True, stream=video_stream
            )
            for pkt in in_container.demux(video_stream):
                if pkt.dts is None:
                    continue
                if pkt.dts > end_ts:
                    break
                if pkt.dts >= start_ts:
                    pkt.stream = out_stream
                    out_container.mux(pkt)


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
        (base / "trimmed_videos").mkdir(exist_ok=True)
        with open(base / "annotation" / "detection_annotations.json") as fd:
            for key, value in tqdm.tqdm(json.load(fd).items()):
                annotations = [
                    Annotation(
                        label=ann["label"],
                        segment=tuple(float(t) for t in ann["segment"]),
                    )
                    for ann in value.pop("annotations")
                ]
                taxonomy = value.pop("taxnomy")
                det = Detection(
                    id=key, taxonomy=taxonomy, annotations=annotations, **value
                )

                if len(det.annotations) > 1:
                    for k, ann in enumerate(det.annotations):
                        _trim(
                            base / "full_videos" / f"{det.id}.mp4",
                            base / "trimmed_videos" / f"{det.id}_{k + 1}.mp4",
                            *ann.segment,
                        )
                else:
                    ann = det.annotations[0]
                    _trim(
                        base / "full_videos" / f"{det.id}.mp4",
                        base / "trimmed_videos" / f"{det.id}.mp4",
                        *ann.segment,
                    )


if __name__ == "__main__":
    main(tyro.cli(Args))
