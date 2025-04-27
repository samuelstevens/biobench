import pathlib
import sys

import beartype


@beartype.beartype
def get_python_files(paths: list[str]) -> list[pathlib.Path]:
    """Get all Python files from a list of paths.

    If a path is a directory, recursively find all Python files within it. If a path is a file, include it only if it's a Python file.

    Args:
        paths: List of file or directory paths to process.

    Returns:
        List of Path objects for Python files.
    """
    # Change this to a generator that yields paths to reduce total memory usage. Use collections.abc to properly type the signature. AI!
    python_files = []
    for path_str in paths:
        path = pathlib.Path(path_str)
        if path.is_dir():
            python_files.extend(path.rglob("*.py"))
        elif path.is_file() and path.suffix == ".py":
            python_files.append(path)
    return python_files


@beartype.beartype
def main(in_paths: list[str]) -> int:
    """Check Python files for non-ASCII characters.

    Recursively scans all Python files in the given directories and checks if they
    contain any non-ASCII characters. Prints a list of files with non-ASCII characters
    or an "all clear" message to stderr if none are found.

    Args:
        in_paths: List of paths (files or directories) to scan.

    Returns:
        int: 0 if all files contain only ASCII characters, 1 otherwise.
    """
    failed = []
    for py in get_python_files(in_paths):
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
