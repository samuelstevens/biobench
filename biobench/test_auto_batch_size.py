import itertools
import math
import threading
import typing

import pytest
import torch
import torch.utils.data

from . import helpers


def make_dataloader(size: int):
    data = torch.rand(size, 3, 32, 32)
    ds = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)


def make_probe(max_ok: int, *, style: typing.Literal["oom", "sdpa"] = "oom"):
    """
    Probe that raises a CUDA-style OOM if the *batch* is larger than `max_ok`. Works on CPU; keeps test independent of real GPU RAM.

    -1 indicates to never OOM.
    """

    def probe(batch):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        if max_ok > 0 and imgs.shape[0] > max_ok:
            if style == "oom":
                raise RuntimeError("CUDA out of memory.")  # substring preserved
            elif style == "sdpa":
                raise RuntimeError("CUDA error: invalid configuration argument")
            else:
                typing.assert_never(style)
        return imgs.mean()  # cheap op

    return probe


def test_ctx_runs_and_restores():
    dataloader = make_dataloader(128)
    orig = dataloader.batch_sampler.batch_size
    probe = make_probe(max_ok=8)  # fail when >8

    with helpers.auto_batch_size(dataloader, probe=probe, schedule=(2, 4, 8, 16)):
        # context executes arbitrary user code w/out throwing
        for _ in itertools.islice(dataloader, 3):
            pass
        # inside ctx largest successful should be 8
        assert dataloader.batch_sampler.batch_size == 8

    # after exit, original value restored
    assert dataloader.batch_sampler.batch_size == orig


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="only meaningful on a GPU box"
)
def test_with_gpu():
    """Smoke-test true GPU execution (no artificial OOM)."""

    dataloader = make_dataloader(128)
    with helpers.auto_batch_size(dataloader, probe=lambda x: x * 1, schedule=(2, 4)):
        next(iter(dataloader))


def test_len_adjusts_with_batch_size():
    """
    `len(dataloader)` reflects the tuned batch size inside the CM and restores afterward.
    """
    dataloader = make_dataloader(30)
    n_samples = len(dataloader.dataset)  # 30
    orig_bs = dataloader.batch_sampler.batch_size  # 2
    assert len(dataloader) == math.ceil(n_samples / orig_bs)  # 15

    probe = make_probe(max_ok=8)  # tuning will settle on 8

    with helpers.auto_batch_size(dataloader, probe=probe, schedule=(2, 4, 8, 16)):
        tuned_bs = dataloader.batch_sampler.batch_size
        assert tuned_bs == 8

        inside_len = len(dataloader)
        assert inside_len == math.ceil(n_samples / tuned_bs)  # 4

    # back to original
    assert dataloader.batch_sampler.batch_size == orig_bs
    assert len(dataloader) == math.ceil(n_samples / orig_bs)  # 15


def test_invalid_cfg_error_handled():
    """
    Helper treats 'invalid configuration argument' the same as OOM.
    """
    dataloader = make_dataloader(128)
    orig = dataloader.batch_sampler.batch_size
    probe = make_probe(max_ok=4, style="sdpa")  # fail when >4

    with helpers.auto_batch_size(dataloader, probe=probe, schedule=(2, 4, 8)):
        # should settle on 4
        assert dataloader.batch_sampler.batch_size == 4

    # restored afterwards
    assert dataloader.batch_sampler.batch_size == orig


def test_terminates_on_short_dataset():
    """
    If dataset has < tried batch-size, auto_batch_size must still terminate.
    """

    dataloader = make_dataloader(12)

    def run():
        with helpers.auto_batch_size(dataloader, probe=make_probe(-1)):
            pass  # nothing

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=2.0)  # 2-second cap
    assert not t.is_alive(), "auto_batch_size never terminated on tiny dataset"


def test_caps_at_dataset_len_generic():
    class Sample(typing.NamedTuple):
        img: torch.Tensor
        meta: dict[str, int]

    def make_dataset_namedtuple(n):
        data = torch.rand(n, 3, 32, 32)
        ds = [Sample(img=x, meta={"idx": i}) for i, x in enumerate(data)]
        return ds

    ds = make_dataset_namedtuple(12)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    with helpers.auto_batch_size(
        loader, probe=lambda b: b.img.mean(), schedule=(4, 8, 16, 32)
    ):
        assert loader.batch_sampler.batch_size == len(ds)  # 12


def test_upper_caps_batch_size():
    """
    With upper=32 the helper must stop at 32 even though larger sizes fit.
    """

    dataloader = make_dataloader(128)
    with helpers.auto_batch_size(
        dataloader, probe=make_probe(-1), schedule=(2, 4, 8, 16, 32, 64, 128), upper=32
    ):
        assert dataloader.batch_sampler.batch_size == 32, "should cap at upper"


def test_upper_below_start_allowed():
    """
    If upper < initial batch size, helper should *lower* to upper immediately.
    """
    dataloader = make_dataloader(128)
    with helpers.auto_batch_size(
        dataloader, probe=make_probe(-1), schedule=(2, 4, 8), upper=1
    ):
        assert dataloader.batch_sampler.batch_size == 1


def _probe_flaky(flake_at: int):
    """Calls above flake_at work once, but not again."""
    seen = set()

    def inner(batch):
        bsz = helpers.infer_batch_size(batch)
        if bsz not in seen:
            seen.add(bsz)

        if bsz not in seen or bsz < flake_at:
            return batch[0].mean()

        raise RuntimeError("CUDA out of memory.")

    return inner


def test_backoff_after_flaky_probe():
    # 8 succeeds then OOMs; helper should fall back to 6.
    dataloader = make_dataloader(128)
    with helpers.auto_batch_size(
        dataloader, probe=_probe_flaky(flake_at=8), schedule=(2, 3, 4, 6, 8, 12, 16)
    ):
        assert dataloader.batch_sampler.batch_size == 6
