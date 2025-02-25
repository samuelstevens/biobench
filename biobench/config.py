import dataclasses
import os.path
import typing

# @dataclasses.dataclass(frozen=True)
# class ModelArgsCvml:
#     org: str
#     ckpt: str
#     def to_dict(self) -> dict[str, object]:
#         return {"type": "cvml", **dataclasses.asdict(self)}


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    model_org: str
    model_ckpt: str

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
    prompts: typing.Literal["single", "multi"] = "single"
    cot_enabled: bool = False
    parallel: int = 1
    """Number of parallel requests per second to MLLM service providers."""

    # CVML only

    # Task-specific args
    ages_data: str = ""
    # Add a *_data field for each task. AI!

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)
