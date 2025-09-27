# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wilds",
#     "tyro",
# ]
# ///
import dataclasses

import tyro
import wilds


@dataclasses.dataclass(frozen=True)
class Args:
    dir: str = "."
    """where to save data."""
    download: bool = True
    """whether to download the data."""


def main(args: Args):
    wilds.get_dataset(dataset="iwildcam", download=args.download, root_dir=args.dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
