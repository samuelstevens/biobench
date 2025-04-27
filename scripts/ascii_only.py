import collections.abc
import pathlib
import sys

import beartype


@beartype.beartype
def get_python_files(paths: list[str]) -> collections.abc.Iterator[pathlib.Path]:
    """Get all Python files from a list of paths.

    If a path is a directory, recursively find all Python files within it. If a path is a file, include it only if it's a Python file.

    Args:
        paths: List of file or directory paths to process.

    Returns:
        Iterator of Path objects for Python files.
    """
    for path_str in paths:
        path = pathlib.Path(path_str)
        if path.is_dir():
            yield from path.rglob("*.py")
        elif path.is_file() and path.suffix == ".py":
            yield path


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
        try:
            with open(py, "r", encoding="ascii") as f:
                for line_num, line in enumerate(f, 1):
                    pass  # Just checking if any line raises an exception
        except UnicodeDecodeError as e:
            # Get the problematic line and character
            with open(py, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Extract error details
            start = e.start
            end = e.end
            bad_byte = e.object[start:end]
            line_num = 1
            char_pos = start

            # Find the line number and character position
            for i, line in enumerate(lines):
                if char_pos < len(line):
                    line_num = i + 1
                    break
                char_pos -= len(line)

            # Get the problematic line
            problem_line = lines[line_num - 1].rstrip("\n")

            # Create a pointer to the problematic character
            pointer = " " * char_pos + "^"

            print(f"{py}:{line_num}:{char_pos + 1}: Non-ASCII character detected")
            print(f" {problem_line}")
            print(f" {pointer}")
            print(f" Problematic bytes: {bad_byte!r}")
            print()

            failed.append(py)

    if failed:
        return len(failed)

    print("All clear: no non-ASCII characters found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import tyro

    sys.exit(tyro.cli(main))
