import dataclasses
import os.path
import tomllib
import typing


@dataclasses.dataclass(frozen=True)
class Model:
    method: typing.Literal["cvml", "mllm"]
    org: str
    ckpt: str


@dataclasses.dataclass(frozen=True)
class Newt:
    """Configuration options specific to the NeWT benchmark."""

    # Filter options - can specify task names, clusters, sub-clusters, or combinations
    tasks: list[str] | None = None
    """List of specific NeWT task names to run. If None, all tasks are included unless filtered by other criteria."""

    include_clusters: list[str] | None = None
    """List of NeWT task clusters to run (e.g., "appearance", "behavior", "context"). If None, all clusters are included."""

    include_subclusters: list[str] | None = None
    """List of NeWT task sub-clusters to run (e.g., "species", "age", "health"). If None, all sub-clusters are included."""

    exclude_tasks: list[str] | None = None
    """List of task names to exclude even if they match other criteria."""

    exclude_clusters: list[str] | None = None
    """List of cluster names to exclude even if they contain tasks that match other criteria."""

    exclude_subclusters: list[str] | None = None
    """List of sub-cluster names to exclude even if they contain tasks that match other criteria."""


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
    batch_size: int = 256
    """Batch size for computer vision model."""
    n_workers: int = 4
    """Number of dataloader worker processes."""
    seed: int = 17
    """Radnom seed."""

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

    newt: Newt = dataclasses.field(default_factory=Newt)

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def load(path: str) -> list[Experiment]:
    """
    Load experiments from a TOML file.

    None of the fields in Experiment are lists, so anytime we find a list in the TOML, we add another dimension to our grid search over all possible experiments.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"TOML file {path} must contain a dictionary at the root level"
        )

    # Extract models list
    models = data.pop("models", [])
    if not isinstance(models, list):
        raise ValueError("models must be a list of tables in TOML")

    # Start with models as base experiments
    experiments = [{"model": Model(**model)} for model in models]

    # Handle NeWT config specially
    newt = data.pop("newt", {})

    # For each remaining field in the TOML
    for key, value in data.items():
        new_experiments = []

        # Convert single values to lists
        if not isinstance(value, list):
            value = [value]

        # For each existing partial experiment
        for exp in experiments:
            # Add every value for this field
            for v in value:
                new_exp = exp.copy()
                new_exp[key] = v
                new_experiments.append(new_exp)

        experiments = new_experiments

    # Now add the NeWT config to all experiments
    for exp in experiments:
        exp["newt"] = Newt(**newt)

    # Convert dictionaries to Experiment objects
    return [Experiment(**exp) for exp in experiments]
