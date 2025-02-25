import dataclasses
import os.path
import typing


@dataclasses.dataclass(frozen=True)
class Model:
    org: str
    ckpt: str


@dataclasses.dataclass(frozen=True)
class Experiment:
    model: Model

    n_train: int = -1
    """Number of maximum training samples. Negative number means use all of them."""
    n_test: int = -1
    """Number of test samples. Negative number means use all of them."""
    sampling: typing.Literal["uniform", "balanced"] = "uniform"

    device: typing.Literal["cpu", "mps", "cuda"] = "cuda"
    """which kind of accelerator to use."""
    debug: bool = False
    """whether to run in debug mode."""

    ssl: bool = True
    """Use SSL when connecting to remote servers to download checkpoints; use --no-ssl if your machine has certificate issues. See `biobench.third_party_models.get_ssl()` for a discussion of how this works."""
    # Reporting and graphing.
    report_to: str = os.path.join(".", "results")
    """where to save reports to."""
    graph: bool = True
    """whether to make graphs."""
    graph_to: str = os.path.join(".", "graphs")
    """where to save graphs to."""
    log_to: str = os.path.join(".", "logs")
    """where to save logs to."""

    # MLLM only
    temp: float = 0.0
    prompting: typing.Literal["single", "multi"] = "single"
    cot_enabled: bool = False
    parallel: int = 1
    """Number of parallel requests per second to MLLM service providers."""

    # CVML only
    slurm: bool = False
    """whether to use submitit to run jobs on a slurm cluster."""
    slurm_acct: str = ""
    """slurm account string."""

    # Task-specific args
    ages_data: str = ""
    beluga_data: str = ""
    birds525_data: str = ""
    fishnet_data: str = ""
    imagenet_data: str = ""
    inat21_data: str = ""
    iwildcam_data: str = ""
    kabr_data: str = ""
    leopard_data: str = ""
    newt_data: str = ""
    plankton_data: str = ""
    plantnet_data: str = ""
    rarespecies_data: str = ""

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)
