docs: lint
    uv run pdoc3 --force --html --output-dir docs/html biobench
    uv run pdoc3 --force --output-dir docs/md biobench
    uv run python -m http.server -d docs/html/biobench

types: lint
    uv run pyright biobench

lint: fmt
    ruff check .

fmt:
    isort .
    ruff format .

