# Guide

BioBench aims to make it easy to benchmark many different models on a variety of biology-related computer vision tasks.

Here's how to benchmark a new CLIP model (maybe you developed it?):

0. Clone the biobench repo.
1. Pick your benchmark tasks and download the data.
2. Add your baseline models and your proposed model to a config `.toml` file.
3. Launch one or more benchmark runners with `benchmark.py`.
4. Summarize the results with `report.py`.
5. Analyze your results with notebooks.

Here are each of the steps in more detail.

## Download Benchmark Data

There are 10 tasks in BioBench:

- beluga
- ecdysis
- fishnet
- fungiclef
- herbarium19
- iwildcam
- kabr
- mammalnet
- plankton
- plantnet

<!-- @Claude, can you correct the capitalization and link to the API docs for each task. -->

Each of them is described in their respective docs, linked above. You can download them using the src/biobench/BENCHMARK/download.py script. If you plan to use NeWT (prior work):

```sh
# Remind yourself of the options
uv run src/biobench/newt/download.py --help

# Run the download
uv run src/biobench/newt/download.py --dir "$NFS/datasets/newt" --images --labels
```

Most of the download scripts use a `--dir` arg to describe where to save the dataset.

## Configure Models

Once you have downloaded a dataset, you can configure your master config.

```toml
n_train = -1
report_to = "/local/scratch/$USER/experiments/biobench"

[data]
# Benchmark
beluga = "/research/nfs_su_809/workspace/stevens.994/datasets/beluga"
ecdysis = "/research/nfs_su_809/workspace/stevens.994/datasets/ecdysis"
fishnet = "/research/nfs_su_809/workspace/stevens.994/datasets/fishnet"
fungiclef = "/research/nfs_su_809/workspace/stevens.994/datasets/fungiclef"
herbarium19 = "/research/nfs_su_809/workspace/stevens.994/datasets/herbarium19"
iwildcam = "/research/nfs_su_809/workspace/stevens.994/datasets/iwildcam"
kabr = "/research/nfs_su_809/workspace/stevens.994/datasets/kabr"
mammalnet = "/research/nfs_su_809/workspace/stevens.994/datasets/mammalnet"
plankton = "/research/nfs_su_809/workspace/stevens.994/datasets/plankton"
plantnet = "/research/nfs_su_809/workspace/stevens.994/datasets/plantnet"

[[models]]
org = "open-clip"
ckpt = "ViT-SO400M-16-SigLIP2-512/webli"

[[models]]
org = "open-clip"
ckpt = "PE-Core-L-14-336/meta"

[[models]]
org = "open-clip"
ckpt = "hf-hub:imageomics/bioclip-2"

[[models]]
org = "open-clip"
ckpt = "hf-hub:imageomics/bioclip-2.5-vith14"

[[models]]
org = "dinov3"
ckpt = "/research/nfs_su_809/workspace/stevens.994/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

##############
# YOUR MODEL #
##############

[[models]]
org = "open-clip"
ckpt = "hf-hub:imageomics/bioclip-2.5-vith14"
```

## Launch Runners

## Report

## Analyze with Notebooks
