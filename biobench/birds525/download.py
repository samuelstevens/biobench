import dataclasses

import beartype
import tyro


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    write_to: str = "."
    """where to save the data."""


@beartype.beartype
def main(args: Args):
    print("Run:")
    print()
    print(
        f"  kaggle datasets download gpiosenka/100-bird-species --path {args.write_to}"
    )
    print()
    print("Then run:")
    print()
    print(f"  cd '{args.write_to}' && unzip 100-bird-species.zip")
    print()
    print()


if __name__ == "__main__":
    main(tyro.cli(Args))
