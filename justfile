docs: fmt
    yek biobench *.py *.md > docs/llms.txt
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True biobench benchmark

test: fmt
    uv run pytest --cov biobench --cov-report term --cov-report json --json-report --json-report-file pytest.json -n 32 biobench || true
    uv run coverage-badge -o docs/coverage.svg -f
    uv run scripts/regressions.py

types: lint
    uv run pyright biobench

lint: fmt
    uv run ruff check --fix biobench benchmark.py

fmt:
    uv run ruff format --preview .

