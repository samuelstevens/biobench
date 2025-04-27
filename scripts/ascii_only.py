def main(root):
    # Add a docstring to this function. The point is to check if any files have non ASCII chars and print a list of the files at the end, or on stderr an "all clear" message. The return code should be non-zero if it failed. AI!
    failed = []
    for py in root.rglob("*.py"):
        txt = py.read_bytes()
        try:
            txt.decode("ascii")
        except UnicodeDecodeError as e:
            print(f"{py} contains non-ASCII: {e}")
            failed.append(py)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
