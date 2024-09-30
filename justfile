docs: lint
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True biobench benchmark

types: lint
    uv run pyright biobench

lint: fmt
    ruff check biobench benchmark.py

fmt:
    isort .
    ruff format --preview .

