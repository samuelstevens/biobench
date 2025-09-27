# Testing

**Session‑scoped, parametrised fixture** is the centrepiece of out tests.

## TL;DR

We turn the checkpoint list into a **parametrised fixture** so that pytest automatically generates an independent test item for every `(test‑function, checkpoint)` pair, caches the expensive model‑load once per checkpoint for the whole session, and gives us clean, filterable IDs in the test report. This is the idiomatic way to fan‑out identical assertions over multiple datasets or configurations, and it avoids manual loops or repeated downloads.

## Separate test items & readable reports
`@pytest.fixture(..., params=CKPTS)` makes pytest expand each consuming test into *N* items (one per checkpoint). The resulting IDs appear as `test_same_shape_single[apple/aimv2‑large…]`, which pinpoints failures and lets you run a single case with `-k`.

## Automatic caching
With `scope="session"` the fixture is created once per parameter and reused everywhere. Heavy objects (two models and their weights) load exactly one time, saving minutes of startup.

## First‑class pytest features
Because each checkpoint is a real test item you can mark or skip it (`pytest.param(..., marks=...)`) or xfail just one checkpoint. This is impossible if you write an inner `for`‑loop instead.

## Composability
Other fixtures (e.g. a random image generator) can depend on `models`; they inherit the same parameter seamlessly without extra wiring.

## Why *session* scope?

* **Most expensive setup wins**: loading two ViT‑L checkpoints dwarfs per‑test compute, so widest scope gives biggest pay‑off.
* **Stateless models**: evaluation‑mode transformers are read‑only, so sharing them across tests is safe; there’s no gradient accumulation or RNG mutation.
* **Disk cache friendliness**: the `cache_dir` argument points to your global HF cache, so downloads happen once even across separate pytest sessions.


## Quick mental model

1. **Fixture creation**
   * Pytest iterates over `CKPTS`; for each, runs the body once, caches the `(hf,bio)` pair.
2. **Test expansion**
   * For every test that requests `models`, pytest clones it per checkpoint, injecting the cached pair.
3. **Execution order**
   * Fixture builds happen before the first test needing them. If ten tests all use `models`, the models load only once per checkpoint.
4. **Random helper `_rand()`**
   * Generates deterministic tensors (seed=0) so the value comparison is reproducible.

Keep this in mind and the tests should still make sense even after a long hiatus.
