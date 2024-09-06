Module benchmark
================
Entrypoint for running all benchmarks.

# Tutorial

This is a short tutorial on using `biobench` to answer a simple research question.

**Research question: Why does [DINOv2](https://github.com/facebookresearch/dinov2) outperform [BioCLIP](https://imageomics.github.io/bioclip/) on classification tasks?**

To answer this question, we need to:

1. Evaluate both models on all tasks in `biobench` using the `benchmark.py` script.
2. Compare mean scores across all classification tasks for both models.
3. Look at specific examples that DINOv2 correctly classifies and BioCLIP incorrectly classifies.

The `biobench` package will completely do #1, #2, and makes it easy to do #3.

## 1. Get Results

First, run:

```sh
python benchmark.py \
  --jobs none \
  --kabr-args.dataset-dir /local/scratch/stevens.994/datasets/KABR/ \
  --newt-args.dataset-dir /local/scratch/stevens.994/datasets/newt/ \
  --model.ckpt hf-hub:imageomics/bioclip
```

In practice, you probably want some combination of the following:

```sh
CUDA_VISIBLE_DEVICES=1 uv run python benchmark.py \
  --jobs none \
  --kabr-args.dataset-dir /local/scratch/stevens.994/datasets/KABR/ \
  --newt-args.dataset-dir /local/scratch/stevens.994/datasets/newt/ \
  --model.ckpt hf-hub:imageomics/bioclip
```

Where you combine `CUDA_VISIBLE_DEVICES` and `uv run` (since we all use `uv` now) with the actual call to Python.

Then we need to evaluate DINOv2.
DINOv2 is suppported using the `TimmVit` integration in `third_party.py`.
We're interested in the best ViT-B model, which is `vit_base_patch14_reg4_dinov2.lvd142m` (ViT-B/14 with 4 registers).

```sh
python benchmark.py \
  --jobs none \
  --kabr-args.dataset-dir /local/scratch/stevens.994/datasets/KABR/ \
  --newt-args.dataset-dir /local/scratch/stevens.994/datasets/newt/ \
  --model.org timm-vit --model.ckpt vit_base_patch14_reg4_dinov2.lvd142m
```

The same caveats apply as above; you might need to specify CUDA_VISIBLE_DEVICES.

After running these scripts, you will have some results stored in `results/` that we can analyze with the starter notebooks.

## 2. View Results

These notebooks are in `notebooks/`.
Open the notebook `notebooks/tutorial.py` using

```sh
marimo edit
```

Functions
---------

`main(args: benchmark.Args)`
:   

`save(args: benchmark.Args, report: biobench.interfaces.BenchmarkReport) ‑> None`
:   Saves the report to disk in a machine-readable JSON format.

Classes
-------

`Args(jobs: Literal['slurm', 'host', 'none'] = 'none', model_org: Literal['timm-vit', 'open_clip'] = 'open_clip', model_ckpt: str = 'RN50/openai', device: Literal['cpu', 'cuda'] = 'cuda', newt_run: bool = True, newt_args: biobench.newt.Args = <factory>, kabr_run: bool = True, kabr_args: biobench.kabr.Args = <factory>, plantnet_run: bool = True, plantnet_args: biobench.plantnet.Args = <factory>, report_to: str = './reports')`
:   Params to run one or more benchmarks in a parallel setting.

    ### Class variables

    `device: Literal['cpu', 'cuda']`
    :   which kind of accelerator to use.

    `jobs: Literal['slurm', 'host', 'none']`
    :   what kind of jobs we should use for parallel processing: slurm cluster, multiple processes on the same machine, or just a single process.

    `kabr_args: biobench.kabr.Args`
    :   arguments for the KABR benchmark.

    `kabr_run: bool`
    :   whether to run the KABR benchmark.

    `model_ckpt: str`
    :

    `model_org: Literal['timm-vit', 'open_clip']`
    :   Where to load models from.

    `newt_args: biobench.newt.Args`
    :   arguments for the NeWT benchmark.

    `newt_run: bool`
    :   whether to run the NeWT benchmark.

    `plantnet_args: biobench.plantnet.Args`
    :   arguments for the plantnet benchmark.

    `plantnet_run: bool`
    :   whether to run the Pl@ntNet benchmark.

    `report_to: str`
    :   where to save reports to.

    ### Methods

    `report_path(self, report: biobench.interfaces.BenchmarkReport) ‑> str`
    :

`DummyExecutor()`
:   Dummy class to satisfy the Executor interface. Directly runs the function in the main process for easy debugging.

    ### Ancestors (in MRO)

    * concurrent.futures._base.Executor

    ### Methods

    `submit(self, fn, /, *args, **kwargs)`
    :   runs `fn` directly in the main process and returns a `concurrent.futures.Future` with the result.
        
        Returns: