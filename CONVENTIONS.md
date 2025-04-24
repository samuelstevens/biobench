Imports: e.g. `import numpy as np`, `import polars as pl`, then always refer to `np.`, `pl.` rather than `from X import *`.

Polars instead of Pandas.

Type annotations & runtime checks

  - Every public function or class uses `@beartype.beartype`.
  - Tensor‑shapes are decorated with `@jaxtyped(typechecker=beartype.beartype)` and static typing via `jaxtyping`.
  - Data‐holding classes (e.g. `Config`, task `Args`) are `@dataclasses.dataclass(frozen=True)`.

Naming

- Classes: `CamelCase` (e.g. `Dataset`, `Args`, `Features`).
- Functions & variables: `snake_case` (e.g. `download_split`, `md5_of_file`, `chunk_size_kb`).
- Constants: `UPPER_SNAKE` at module top (e.g. `URLS = {...}`).

Misc

- Use `helpers.progress(...)` for loop progress instead of raw `tqdm`.
- Module‐level `logger = logging.getLogger(__name__)`, then `logger.info/warning`.

Download scripts
- Header block with `/// script` and a `dependencies = [...]` list for self‐documenting prerequisites.
- Chunked streaming via `requests.get(..., stream=True)` + `tqdm` progress bar.
- Checksum verification before extraction.

File layout & naming
- One task per folder (`biobench/herbarium19/`) with `download.py` and `__init__.py`.
- The download script, then the task module implementing `benchmark(cfg)`.

Testing style
- Pytest with fixtures & parameterization.
- Hypothesis in helpers tests.
