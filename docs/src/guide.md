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

- [`biobench.beluga`][]
- [`biobench.ecdysis`][]
- [`biobench.fishnet`][]
- [`biobench.fungiclef`][]
- [`biobench.herbarium19`][]
- [`biobench.iwildcam`][]
- [`biobench.kabr`][]
- [`biobench.mammalnet`][]
- [`biobench.plankton`][]
- [`biobench.plantnet`][]

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
report_to = "$SCRATCH/$USER/biobench"

[data]
newt = "$NFS/datasets/newt"

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

Here's what this means:

```toml
n_train = -1
```

Use all training samples. If you were interested in sample-efficient learning, you might set this to `n_train = 10` or similar.

```toml
report_to = "/$SCRATCH/$USER/biobench"
```

All results will be written to `/$SCRATCH/$USER/biobench/reports.sqlite`

!!! warning

    You cannot write results to a SQLite file on an NFS drive. If you truly have exactly one runner in parallel, this is okay, but multiple parallel runners can corrupt SQLite files when it's on an NFS drive.

```toml
[data]
newt = "$NFS/datasets/newt"
```

Reference the benchmarks that you want to run, and provide a path.

```toml
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
org = "timm"
ckpt = "convnext_tiny.in12k"

[[models]]
org = "dinov3"
ckpt = "/PATH/TO/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
```

These are your baseline models. Many of the strongest vision encoders are compatible with either [`timm`](https://github.com/huggingface/pytorch-image-models) or [`open_clip`](https://github.com/mlfoundations/open_clip) and can be trivially loaded via a `ckpt` slug.

For instance, the BioCLIP 2 model references this snippet of code on the [HuggingFace model page](https://huggingface.co/imageomics/bioclip-2?library=open_clip):

```python
import open_clip

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
```

You can pass `"hf-hub:imageomics/bioclip-2"` as the `ckpt` field.


!!! note

    DINOv3 requires local checkpoints to be downloaded, because the weights are not publicly available without using a HF token of some kind.

```toml
##############
# YOUR MODEL #
##############

[[models]]
org = "open-clip"
ckpt = "hf-hub:imageomics/bioclip-2.5-vith14"
```

Perhaps you are developing a new version of BioCLIP. If it's published on HuggingFace already, you can use the `hf-hub` trick as before.
If not, the `open_clip` org also accepts local paths like `local:ViT-L-14//models/vit-l-14-tol200m-laion2b-replay-ep30.pt` (see the snippet in third_party_models.py under `elif ckpt.startswith("local:"):`).

See the [`biobench.third_party_models.OpenClip`][] API docs for more details.

If your model is not compatible with `timm` or `open_clip`, you can also add a new model class by subclassing [`biobench.registry.VisionBackbone`][]. See the existing implementations in [`biobench.aimv2`][], [`biobench.vjepa`][], or [`biobench.third_party_models`][] for examples.

## Launch Runners

Once your config is set up, you can launch one or more benchmark runner processes that will parse your config and run a sequence of benchmark tasks.
Runners are cooperative and use a SQLite database to coordinate jobs. 
I typically run something like:

```sh
# In my first shell
CUDA_VISIBLE_DEVICES=0 uv run benchmark.py --cfgs configs/my-config.toml

# In a second shell, starting at least 10 or so seconds afterwards to avoid deadlocks.
CUDA_VISIBLE_DEVICES=1 uv run benchmark.py --cfgs configs/my-config.toml
```

This runs two jobs on separate GPUs.
They'll coordinate, and not re-run any already benchmarked model/task combinations.

## Report

After your jobs are done, you can run `uv run report.py --db $SCRATCH/$USER/biobench/reports.sqlite` which will do some statistical significance testing with bootstrapping to compute statistically significant improvements.

This will produce [`results.json`](https://github.com/Imageomics/biobench/blob/main/docs/data/results.json).

## Analyze with Notebooks

However, for intermediate analysis, I strongly recommend combining Python notebooks like [JupyterLab](https://jupyter.org/) or [marimo](https://marimo.io/) with the SQLite database via [DuckDB](https://duckdb.org/) or [Polars](https://pola.rs/) to make charts and tables as necessary.
