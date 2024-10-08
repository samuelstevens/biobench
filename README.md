# Biology Benchmark (`biobench`)

This library is an easy-to-read benchmark for biology-related computer vision tasks.

It aims to make it easy to:

1. Evaluate new models.
2. Add new tasks.
3. Understand meaningful (or not) differences in model performance.

## Getting Started

I use [uv](https://docs.astral.sh/uv/) for Python which makes it easy to manage Python versions, dependencies, virtual environments, etc.

To install uv, run `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Then download at least one of the dataset.
NeWT is really easy to download.

```sh
uv run biobench/newt/download.py --dir ./newt
```

Download it wherever you want on your own filesystem.

Then run just the NeWT benchmark on all the default models.

```sh
CUDA_VISIBLE_DEVICES=0 uv run benchmark.py \
  --newt-run --newt-args.datadir ./newt
```


## Concrete Goals

*Easy*, *fast*, *reproducible*, *understandable* evaluation of PyTorch computer vision models across a suite of biology-related vision tasks.

- *Easy*: one launch script, with all options documented in the code and in auto-generated web documentation.
- *Fast*: Each evaluation takes at most 1 hour of A100 or A6000 time. There might be $n$ evaluations, so $n$ hours of A100, but it is embarrassingly parallel and the launch script supports easy parallel running and reporting.
- *Reproducible*: the results include instructions to regenerate these results from scratch, assuming access to the `biobench` Git repo and that web dependencies have not changed.[^web-deps]
- *Understandable*: results are in a machine-readable format, but include a simple human-readable notebook for reading. Common analyses (mean score across all tasks) are included in the notebook and take under one second to run.

[^web-deps]: Web dependencies include things like datasets being available from their original source, Huggingface datasets can be re-downloaded, model checkpoints do not change, etc.

## Road Map

1. Add 5-shot RareSpecies with simpleshot (like in BioCLIP paper). This is blocked because the Huggingface dataset doesn't work ([see this issue](https://huggingface.co/datasets/imageomics/rare-species/discussions/8)).
2. Update docs. I will do this during travel, then push when I reconnect to the network.
3. Change Pl@ntNet to account for large class imbalance in training data.
4. Add FishVista for trait prediction. This is another non-classification task, and we are specifically interested in traits. But it will take more work because we have to match bounding boxes and patch-level features which is challenging after resizes.
