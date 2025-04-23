import itertools
import pathlib
import textwrap

from . import config


def _write_toml(tmp_path: pathlib.Path) -> pathlib.Path:
    tmp_path.mkdir(exist_ok=True)
    tmp_file = tmp_path / "grid.toml"
    tmp_file.write_text(
        textwrap.dedent(
            """
            models = [
              { org = "timm",      ckpt = "resnet50" },
              { org = "open-clip", ckpt = "ViT-B-16/openai" },
            ]

            debug   = [ true, false ]
            n_train = [ 10, 100 ]

            [data]
            newt = "/data/newt"
            """
        )
    )
    return tmp_file


def test_load_returns_full_cartesian_product(tmp_path):
    toml_path = _write_toml(tmp_path)
    exps = config.load(str(toml_path))

    # --- expected triples ---
    expected = set(
        itertools.product(
            ["timm", "open-clip"],  # model_org
            [True, False],  # debug
            [10, 100],  # n_train
        )
    )

    # --- actual triples ---
    got = {(exp.model.org, exp.debug, exp.n_train) for exp in exps}

    assert got == expected, (
        f"Missing/extra combinations: expected {expected}, got {got}"
    )

    # data-block propagated
    for exp in exps:
        assert exp.data.newt == "/data/newt"
