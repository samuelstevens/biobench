import pathlib
import sys


def main(in_paths: list[str]) -> int:
    """Check Python files for non-ASCII characters.

    Recursively scans all Python files in the given directory and checks if they contain any non-ASCII characters. Prints a list of files with non-ASCII characters or an "all clear" message to stderr if none are found.

    Args:
        root: Path to the root directory to scan.

    Returns:
        int: 0 if all files contain only ASCII characters, 1 otherwise.
    """
    failed = []
    # If path is a directory, do a glob, otherwise, just check it. Do this as a list[str] -> list[str] helper that takes a list of paths, directories or files, and yields files only. AI!
    for path in in_paths:
        for py in pathlib.Path(path).rglob("*.py"):
            txt = py.read_bytes()
            try:
                txt.decode("ascii")
            except UnicodeDecodeError as e:
                print(f"{py} contains non-ASCII: {e}")
                failed.append(py)

    if failed:
        return 1

    print("All clear: no non-ASCII characters found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import tyro

    sys.exit(tyro.cli(main))
