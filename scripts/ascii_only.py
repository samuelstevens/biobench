def main(root):
    """Check Python files for non-ASCII characters.
    
    Recursively scans all Python files in the given directory and checks if they
    contain any non-ASCII characters. Prints a list of files with non-ASCII characters
    or an "all clear" message to stderr if none are found.
    
    Args:
        root: Path to the root directory to scan.
        
    Returns:
        int: 0 if all files contain only ASCII characters, 1 otherwise.
    """
    failed = []
    for py in root.rglob("*.py"):
        txt = py.read_bytes()
        try:
            txt.decode("ascii")
        except UnicodeDecodeError as e:
            print(f"{py} contains non-ASCII: {e}")
            failed.append(py)
    
    if failed:
        return 1
    else:
        import sys
        print("All clear: no non-ASCII characters found.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    import tyro
    import sys

    sys.exit(tyro.cli(main))
