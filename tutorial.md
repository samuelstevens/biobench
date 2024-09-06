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
