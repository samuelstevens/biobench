Module benchmark
================

Functions
---------

`display(report: biobench.interfaces.BenchmarkReport) ‑> None`
:   

`main(args: benchmark.Args)`
:   

Classes
-------

`Args(jobs: Literal['slurm', 'process', 'none'] = 'none', model: biobench.models.Params = <factory>, device: Literal['cpu', 'cuda'] = 'cuda', newt_run: bool = False, newt_args: biobench.newt.Args = <factory>, kabr_run: bool = False, kabr_args: biobench.kabr.Args = <factory>)`
:   Args(jobs: Literal['slurm', 'process', 'none'] = 'none', model: biobench.models.Params = <factory>, device: Literal['cpu', 'cuda'] = 'cuda', newt_run: bool = False, newt_args: biobench.newt.Args = <factory>, kabr_run: bool = False, kabr_args: biobench.kabr.Args = <factory>)

    ### Class variables

    `device: Literal['cpu', 'cuda']`
    :   which kind of accelerator to use.

    `jobs: Literal['slurm', 'process', 'none']`
    :   what kind of jobs we should use for parallel processing: slurm cluster, multiple processes on the same machine, or just a single process.

    `kabr_args: biobench.kabr.Args`
    :   arguments for the KABR benchmark.

    `kabr_run: bool`
    :   whether to run the KABR benchmark.

    `model: biobench.models.Params`
    :   arguments for the vision backbone.

    `newt_args: biobench.newt.Args`
    :   arguments for the NeWT benchmark.

    `newt_run: bool`
    :   whether to run the NeWT benchmark.

`DummyExecutor()`
:   Dummy class to satisfy the Executor interface. Directly runs the function in the main process for easy debugging.

    ### Ancestors (in MRO)

    * concurrent.futures._base.Executor

    ### Methods

    `submit(self, fn, /, *args, **kwargs)`
    :   runs `fn` directly in the main process and returns a Future with the result.