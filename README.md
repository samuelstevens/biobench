# Biology Benchmark (`biobench`)

This library is an easy-to-read benchmark for biology-related computer vision tasks.

It aims to make it easy to:

1. Evaluate new models.
2. Add new tasks.

## Concrete Goals

*Easy*, *fast*, *reproducible*, *understandable* evaluation of PyTorch computer vision models across a suite of biology-related vision tasks.

- *Easy*: one launch script, with all options documented in the code and in auto-generated web documentation.
- *Fast*: Each evaluation takes at most 1 hour of A100 time. There might be $n$ evaluations, so $n$ hours of A100, but it is embarrassingly parallel and the launch script supports easy parallel running and reporting.
- *Reproducible*: the results include instructions to regenerate these results from scratch, assuming access to the `biobench` Git repo and that web dependencies have not changed.[^web-deps]
- *Understandable*: results are in a machine-readable format, but include a simple human-readable notebook for reading. Common analyses (mean score across all tasks) are included in the notebook and take under one second to run.

[^web-deps]: Web dependencies include things like datasets being available from their original source, Huggingface datasets can be re-downloaded, model checkpoints do not change, etc.


# TODO

## Additional Tasks

* Add Flickr Coyote task
* Add Ohio Small Animals
* Add TreeDetector
* Add Beluga whale re-id
