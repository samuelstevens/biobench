# Examples

## Run Everything

Suppose you want to run all the tasks for all the default models to get started using a local GPU (device 4, for example).
You need to specify all the `--TASK-run` flags and `--TASK-args.datadir` so that each task knows where to load data from.

```sh
CUDA_VISIBLE_DEVICES=4 python benchmark.py \
  --kabr-run --kabr-args.datadir /local/scratch/stevens.994/datasets/kabr \
  --iwildcam-run --iwildcam-args.datadir /local/scratch/stevens.994/datasets/iwildcam \
  --plantnet-run --plantnet-args.datadir /local/scratch/stevens.994/datasets/plantnet \
  --birds525-run --birds525-args.datadir /local/scratch/stevens.994/datasets/birds525 \
  --newt-run --newt-args.datadir /local/scratch/stevens.994/datasets/newt \
  --beluga-run --beluga-args.datadir /local/scratch/stevens.994/datasets/beluga \
  --ages-run --ages-args.datadir /local/scratch/stevens.994/datasets/newt \
  --fishnet-run --fishnet-args.datadir /local/scratch/stevens.994/datasets/fishnet
```

More generally, you can configure options for individual tasks using `--TASK-args.<OPTION>`, which are all documented `python benchmark.py --help`.

## Just One Task

Suppose you just want to run one task (NeWT).

```sh
CUDA_VISIBLE_DEVICES=4 python benchmark.py \
  --newt-run --newt-args.datadir /local/scratch/stevens.994/datasets/newt
```

## Just One Model

Suppose you only want to run the SigLIP SO400M ViT from Open CLIP, but you want to run it on all tasks.
Since that model is a checkpoint in Open CLIP, we can use the `biobench.third_party_models.OpenClip` class to load the checkpoint.

```sh
CUDA_VISIBLE_DEVICES=4 python benchmark.py \
  --kabr-run --kabr-args.datadir /local/scratch/stevens.994/datasets/kabr \
  --iwildcam-run --iwildcam-args.datadir /local/scratch/stevens.994/datasets/iwildcam \
  --plantnet-run --plantnet-args.datadir /local/scratch/stevens.994/datasets/plantnet \
  --birds525-run --birds525-args.datadir /local/scratch/stevens.994/datasets/birds525 \
  --newt-run --newt-args.datadir /local/scratch/stevens.994/datasets/newt \
  --beluga-run --beluga-args.datadir /local/scratch/stevens.994/datasets/beluga \
  --ages-run --ages-args.datadir /local/scratch/stevens.994/datasets/newt \
  --fishnet-run --fishnet-args.datadir /local/scratch/stevens.994/datasets/fishnet \
  --model open-clip ViT-SO400M-14-SigLIP/webli  # <- This is the new line!
```

## Use Slurm

Slurm clusters with lots of GPUs can be used to run lots of tasks in parallel.
It's really easy with `biobench`.

```sh
python benchmark.py \
  --kabr-run --kabr-args.datadir /local/scratch/stevens.994/datasets/kabr \
  --iwildcam-run --iwildcam-args.datadir /local/scratch/stevens.994/datasets/iwildcam \
  --plantnet-run --plantnet-args.datadir /local/scratch/stevens.994/datasets/plantnet \
  --birds525-run --birds525-args.datadir /local/scratch/stevens.994/datasets/birds525 \
  --newt-run --newt-args.datadir /local/scratch/stevens.994/datasets/newt \
  --beluga-run --beluga-args.datadir /local/scratch/stevens.994/datasets/beluga \
  --ages-run --ages-args.datadir /local/scratch/stevens.994/datasets/newt \
  --fishnet-run --fishnet-args.datadir /local/scratch/stevens.994/datasets/fishnet \
  --slurm  # <- Just add --slurm to use slurm!
```

Note that you don't need to specify `CUDA_VISIBLE_DEVICES` anymore because you're not running on the local machine anymore.
