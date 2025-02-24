docs: lint
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True biobench benchmark
    uv run python scripts/docs.py --in-paths biobench benchmark.py --out-fpath docs/llms.txt

types: lint
    uv run pyright biobench

lint: fmt
    ruff check --fix biobench benchmark.py

fmt:
    ruff format --preview .

