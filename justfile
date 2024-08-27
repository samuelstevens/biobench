docs: lint
    uv run pdoc3 --force --html --output-dir docs src.biology_benchmark
    uv run python -m http.server -d docs/src/biology_benchmark

types: lint
    uv run pyright src/biology_benchmark

lint: fmt
    ruff check .

fmt:
    isort .
    ruff format .

