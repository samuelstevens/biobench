docs: lint
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True biobench benchmark
    uv run python scripts/docs.py --in-paths biobench benchmark.py --out-fpath docs/llms.txt

types: lint
    uv run pyright biobench

lint: fmt
    uv run ruff check --fix biobench benchmark.py

fmt:
    uv run ruff format --preview .

