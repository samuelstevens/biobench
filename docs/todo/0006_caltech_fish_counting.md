# 0006. Add Caltech Fish Counting Dataset

Goal: Add a new task `fishcounting` to BioBench, based off of [https://github.com/visipedia/caltech-fish-counting](https://github.com/visipedia/caltech-fish-counting).

Why: Adds another task objective (counting, object tracking) and another modality (underwater sonar cameras).

Done when:

1. `biobench.fishcounting` and `biobench.fishcounting.download` are valid modules in BioBench.
2. `fishcounting` module has the following functions:

```py
@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report: ...
```

```py
@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
```

3. `uv run benchmark.py --cfgs configs/testing.toml` works, where `testing.toml` has the following content:

```toml
n_train = [-1]
report_to = "results-testing"

[data]
fishcounting = "/research/nfs_su_809/workspace/stevens.994/datasets/caltechfishcounting"

[[models]]
org = "open-clip"
ckpt = "ViT-B-32/openai"
```

4. Then also run `uv run report.py` pointing at the test database.

Notes:

Citations for the papers, to be put in the fishcounting docstring:

```bib
@misc{kay2024align,
      title={Align and Distill: Unifying and Improving Domain Adaptive Object Detection}, 
      author={Justin Kay and Timm Haucke and Suzanne Stathatos and Siqi Deng and Erik Young and Pietro Perona and Sara Beery and Grant Van Horn},
      year={2024},
      eprint={2403.12029},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{cfc2022eccv,
    author    = {Kay, Justin and Kulits, Peter and Stathatos, Suzanne and Deng, Siqi and Young, Erik and Beery, Sara and Van Horn, Grant and Perona, Pietro},
    title     = {The Caltech Fish Counting Dataset: A Benchmark for Multiple-Object Tracking and Counting},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2022}
}
```
