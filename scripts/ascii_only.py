import collections.abc
import pathlib
import sys
import re

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


# Common non-ASCII characters and their ASCII replacements
@beartype.beartype
def get_replacement_map() -> dict[str, str]:
    """Get a map of common non-ASCII characters to their ASCII replacements."""
    return {
        # Common typographic characters
        "\u2013": "-",  # en dash
        "\u2014": "--",  # em dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",  # non-breaking space
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2015": "--",  # horizontal bar
        
        # Hyphens and dashes that appear in the errors
        "\u2010": "-",  # hyphen
        "\u2212": "-",  # minus sign
        
        # Accented characters
        "\u00e4": "a",  # ä
        "\u00c4": "A",  # Ä
        "\u00f6": "o",  # ö
        "\u00d6": "O",  # Ö
        "\u00fc": "u",  # ü
        "\u00dc": "U",  # Ü
        "\u00e9": "e",  # é
        "\u00c9": "E",  # É
        "\u00e8": "e",  # è
        "\u00c8": "E",  # È
        "\u00e2": "a",  # â
        "\u00ea": "e",  # ê
        "\u00ee": "i",  # î
        "\u00f4": "o",  # ô
        "\u00fb": "u",  # û
        "\u00e0": "a",  # à
        "\u00e7": "c",  # ç
        "\u00f1": "n",  # ñ
        "\u00e5": "a",  # å
        "\u00c5": "A",  # Å
        
        # Special characters in the errors
        "\u2010\u0038": "-8",  # PEP‑8
    }

@beartype.beartype
def fix_non_ascii(content: str, replacement_map: dict[str, str]) -> str:
    """Replace non-ASCII characters with their ASCII equivalents."""
    for non_ascii, ascii_replacement in replacement_map.items():
        content = content.replace(non_ascii, ascii_replacement)
    return content

@beartype.beartype
def main(in_paths: list[str], fix: bool = False) -> int:
    """Check Python files for non-ASCII characters and optionally fix them.

    Recursively scans all Python files in the given directories and checks if they
    contain any non-ASCII characters. Prints a list of files with non-ASCII characters
    or an "all clear" message to stderr if none are found.

    Args:
        in_paths: List of paths (files or directories) to scan.
        fix: If True, automatically fix common non-ASCII characters.

    Returns:
        int: 0 if all files contain only ASCII characters or were fixed, 1 otherwise.
    """
    replacement_map = get_replacement_map()
    failed = []
    fixed = []
    
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
            
            # Try to fix the file if requested
            if fix:
                # Read the entire file content
                with open(py, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Fix non-ASCII characters
                fixed_content = fix_non_ascii(content, replacement_map)
                
                # Check if we actually fixed anything
                if fixed_content != content:
                    with open(py, "w", encoding="utf-8") as f:
                        f.write(fixed_content)
                    fixed.append(py)
                    continue  # Skip reporting this file as failed
                
            # Create a pointer to the problematic character
            pointer = " " * char_pos + "^"

            print(f"{py}:{line_num}:{char_pos + 1}: Non-ASCII character detected")
            print(f" {problem_line}")
            print(f" {pointer}")
            print(f" Problematic bytes: {bad_byte!r}")
            print()

            failed.append(py)

    if fixed:
        print(f"Fixed {len(fixed)} files:", file=sys.stderr)
        for py in fixed:
            print(f"  {py}", file=sys.stderr)
        print(file=sys.stderr)
    
    if failed:
        print(f"Failed to fix {len(failed)} files:", file=sys.stderr)
        for py in failed:
            print(f"  {py}", file=sys.stderr)
        return len(failed)

    print("All clear: no non-ASCII characters found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import tyro

    sys.exit(tyro.cli(main))
