# CONTRIBUTING

## 1. TL;DR

Install [uv](https://docs.astral.sh/uv/).
Clone this repository, then from the root directory:

```sh
uv run python -m saev --help
```

You also need [yek](https://github.com/bodo-run/yek) and [lychee](https://github.com/lycheeverse/lychee) for generating docs.

If you want to do any of the web interface work, you need [elm](https://guide.elm-lang.org/install/elm.html), [elm-format](https://github.com/avh4/elm-format/releases/latest), [tailwindcss](https://github.com/tailwindlabs/tailwindcss/releases/latest) and [bun](https://bun.sh/).

## 2. Repo Layout

```
benchmark.py  <- Launch script for benchmarking models on tasks.
report.py     <- Launch script for producing the report.json file.
biobench/     <- Source code for benchmarking.
  beluga/
    __init__.py
    download.py
  fishnet/
    __init__.py
    download.py
  ...
  config.py
  registry.py
  reporting.py
  schema.sql
  third_party_models.py
docs/
  api/
  assets/
  data/
  research/
  todo/
  index.html
web/
  src/
    Leaderboard.elm
  index.html
  main.css
```

## 3. Coding Style & Conventions

See [AGENTS.md](AGENTS.md).

## 4. Testing & Linting

1. Run `just test`.
2. Check that there are no regressions. Unless you are certain tests are not needed, the coverage % should either stay the same or increase.
3. Run `just docs`.
4. Fix any missing doc links.

## 5. Code of Conduct

Be polite, kind and assume good intent.
