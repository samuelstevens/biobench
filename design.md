# Design Notes

I looked into running multiple benchmarks in parallel on the same machine with different GPUs.
I spent a lot of time investigating different ways to make it work, and I discovered several problems:

1. We run out of system memory. My machine is limited not by the number of GPUs, but by the number of dataloaders. With only 1 dataloader per process, I can reliably run 3 jobs in parallel. With 2 dataloaders per process, I get errors when running 3 jobs in parallel.
2. Torch tensors are shared between processes via shared memory, not be value. So after processes would end, you couldn't access the tensors anymore.
3. The APis for multiprocessing are not great. There's `multiprocessing`, `torch.multiprocessing` and `multiprocess` (which uses `dill` in favor of `pickle`).

So, I stuck with single-process jobs and `submitit`.
