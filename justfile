docs: fmt
    rm -rf docs/api
    mkdir -p docs/api
    yek biobench *.py *.md > docs/api/llms.txt
    uv run pdoc3 --force --html --output-dir docs/api --config latex_math=True biobench benchmark report scripts

leaderboard: fmt
    cp web/index.html docs/index.html
    cd web && tailwindcss --input main.css --output ../docs/assets/dist/main.css
    cd web && elm make src/Leaderboard.elm --output ../docs/assets/dist/leaderboard.js --optimize
    bunx --bun uglify-js docs/assets/dist/leaderboard.js --compress 'pure_funcs=[F2,F3,F4,F5,F6,F7,F8,F9,A2,A3,A4,A5,A6,A7,A8,A9],pure_getters,keep_fargs=false,unsafe_comps,unsafe' | bunx --bun uglify-js --mangle --output docs/assets/dist/leaderboard.min.js

test: fmt
    uv run pytest --cov biobench --cov-report term --cov-report json --json-report --json-report-file pytest.json --cov-report html -n 32 biobench || true
    uv run coverage-badge -o docs/assets/coverage.svg -f
    uv run scripts/regressions.py

lint: fmt
    uv run ruff check --fix biobench benchmark.py report.py
    uv run scripts/ascii_only.py --in-paths benchmark.py report.py biobench/ scripts/ --fix

fmt:
    uv run ruff format --preview .
    fd -e elm | xargs elm-format --yes

clean:
    rm -f .coverage*
    rm -f coverage.json
    rm -f pytest.json
    rm -rf .hypothesis/
    rm -rf .ruff_cache/
    rm -rf .pytest_cache/
    rm -rf results-testing/
    rm -rf htmlcov/
