import beartype

from .. import config, reporting


@beartype.beartype
def benchmark(cfg: config.Experiment) -> tuple[config.Model, reporting.Report]:
    raise NotImplementedError()
