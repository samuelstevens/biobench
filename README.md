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


# TODO

## Concrete Steps

- Easy: currently easy to run. I need examples of how to run the script in different contexts, with different goals in mind.
- Fast: With unlimited resources, I can still run the entire benchmark serially in under 1 hour on an A6000. As I add more tasks, I will need to make it more parallel.
- Reproducible: not at all. Right now, results are simply printed to the console with a benchmark name and the score. These results need to be logged to a file that includes instructions on how to re-acquire the results.
- Understandable: results are not in a machine-readable format.

I think my current TODO is making results available in a machine-readable format with instructions on how to reproduce.
This way, I can start to fill in the huge table of MODELS x TASKS to identify trends.
After that, I will continue adding tasks from FGCV11 until it takes more than an hour to run a ViT-B-16 on a single A6000.
Then, I can add parallelization.

## Additional Tasks

- Add TreeDetector
- Add Beluga whale re-id
- Look at FGCV 11 tasks (classification)
- Think about segmentation/object detection tasks
- Provide evidence that most vision tasks are accomplished with $f(img) -> \RR^d$ as an initial step.

