# Biology Benchmark (`biobench`)

This library is an easy-to-read benchmark for biology-related computer vision tasks.

It aims to make it easy to:

1. Evaluate new models.
2. Add new tasks.

## Concrete Goals

*Easy*, *fast*, *reproducible*, *understandable* evaluation of PyTorch computer vision models across a suite of biology-related vision tasks.

- *Easy*: one launch script, with all options documented in the code and in auto-generated web documentation.
- *Fast*: Each evaluation takes at most 1 hour of A100 or A6000 time. There might be $n$ evaluations, so $n$ hours of A100, but it is embarrassingly parallel and the launch script supports easy parallel running and reporting.
- *Reproducible*: the results include instructions to regenerate these results from scratch, assuming access to the `biobench` Git repo and that web dependencies have not changed.[^web-deps]
- *Understandable*: results are in a machine-readable format, but include a simple human-readable notebook for reading. Common analyses (mean score across all tasks) are included in the notebook and take under one second to run.

[^web-deps]: Web dependencies include things like datasets being available from their original source, Huggingface datasets can be re-downloaded, model checkpoints do not change, etc.

# To Do

1. Add RareSpecies with 1-shot and 5-shot variants
2. Host docs on github pages somehow.
3. Make presentation explaining this work to ml-foundations group.

## Concrete Steps

- Easy: currently easy to run. I need examples of how to run the script in different contexts, with different goals in mind.
- Fast: I can still run the entire benchmark serially in under 1 hour on an A6000. As I add more tasks, I will need to make it more parallel. It's possible to run the entire benchmark for one model in under 1 hour on an A6000. However, it's not possible to run it for all four default models. I need to implement `submitit` on OSC.
- Reproducible: Very. However, I need to add instructions on how to reproduce.
- Understandable: results are in a machine-readable format (sqlite database), and produce graphs and a `results.csv` file for quick human viewing. But it would be nice to have a mistake viewer for each task in `notebooks/`.

## Long-Term

- Add Beluga whale re-id
- Add TreeDetector (think about segmentation/object detection tasks)
- Provide evidence that most vision tasks are accomplished with $f(img) -> \mathcal{R}^d$ as an initial step.
- Predict trait/no trait in CUB images---use patch-level features + linear classifier for each possible trait

