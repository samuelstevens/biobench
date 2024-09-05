docs: lint
    uv run pdoc3 --force --output-dir docs/md --config latex_math=True biobench benchmark
    uv run pdoc3 --http :8000 --force --html --output-dir docs/html --config latex_math=True biobench benchmark

types: lint
    uv run pyright biobench

lint: fmt
    ruff check .

fmt:
    isort .
    ruff format --preview .

