docs: lint
    uv run pdoc3 --http :8000 --force --html --output-dir docs --config latex_math=True biobench benchmark

types: lint
    uv run pyright biobench

lint: fmt
    ruff check .

fmt:
    isort .
    ruff format --preview .

