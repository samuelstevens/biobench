docs: fmt
    yek biobench *.py *.md > docs/llms.txt
    rm -rf docs/biobench docs/benchmark.html
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True biobench benchmark report scripts

test: fmt
    uv run pytest --cov biobench --cov-report term --cov-report json --json-report --json-report-file pytest.json --cov-report html -n 32 biobench || true
    uv run coverage-badge -o docs/coverage.svg -f
    uv run scripts/regressions.py

lint: fmt
    uv run ruff check --fix biobench benchmark.py report.py
    uv run scripts/ascii_only.py --in-paths benchmark.py report.py biobench/ scripts/ --fix

fmt:
    uv run ruff format --preview .

clean:
    rm -f .coverage*
    rm -f coverage.json
    rm -f pytest.json
    rm -rf .hypothesis/
    rm -rf .ruff_cache/
    rm -rf .pytest_cache/
    rm -rf results-testing/
    rm -rf htmlcov/
