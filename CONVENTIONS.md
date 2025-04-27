# Code Style

- Keep code simple, explicit, typed, test-driven, and ready for automation.
- Source files are UTF-8 but must contain only ASCII characters. Do not use smart quotes, ellipses, em-dashes, emoji, or other non-ASCII glyphs.
- Docstrings are a single unwrapped paragraph. Rely on your editor's soft-wrap.
- Prefer explicit over implicit constructs. No wildcard imports.

```python
import numpy as np
import polars as pl  # use Polars instead of Pandas
```

Always reference modules by their alias. Never use `from X import *`.

- Decorate every public function or class with `@beartype.beartype`.
- For tensors, add `@jaxtyped(typechecker=beartype.beartype)` and use `jaxtyping` shapes.
- Use frozen `@dataclasses.dataclass` for data containers such as `Config` or `Args`.
- Classes use `CamelCase`, for example `Dataset` or `FeatureExtractor`.
- Functions and variables use `snake_case`, for example `download_split` or `md5_of_file`.
- Constants are `UPPER_SNAKE`, defined at module top, for example `URLS = {...}`.
- File descriptors end in `fd`, for example `log_fd`.
- File paths end in `_fpath`; directories end in `_dpath`.
- Constructors follow verb prefixes:
  - `make_...` returns an object.
  - `get_...` returns a primitive value such as a string or path.
  - `setup_...` performs side effects and returns nothing.

## Shape-suffix notation

Attach suffixes to tensor variables to clarify shape:

- B – batch size
- W – patch grid width
- H – patch grid height
- D – feature dimension
- L – number of latents
- C – number of classes

Example: `acts_BWHD` has shape (batch, width, height, d).

## Logging and progress bars

```python
logger = logging.getLogger(__name__)
logger.info("message")

for x in helpers.progress(dataset):
    ...
```

Use `helpers.progress` instead of `tqdm` so that logging is useful in non-interactive contexts (log files, batch jobs, etc).

# Testing

- Use pytest with fixtures and parameterization.
- Use Hypothesis for property-based tests, especially in helpers.
- Mark slow integration tests with `@pytest.mark.slow`.

# Project layout

Each task lives in its own folder, for example `biobench/herbarium19/`.  
Inside a task folder:

- `download.py` fetches the dataset.
- `__init__.py` exposes the task API, including a `benchmark(cfg)` entry point.

## Download scripts

- Start each downloader with a header line `/// script` and a `dependencies = [...]` list.
- Stream with `requests.get(..., stream=True)` and wrap in `tqdm` for progress.
- Verify checksums before extraction.
