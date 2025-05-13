import dataclasses
import os
import tomllib
import typing

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Model:
    """Configuration for a model to be evaluated.

    This class defines the essential parameters needed to identify and load a specific model for evaluation in the benchmark.

    Attributes:
        org: Organization or source of the model (e.g., "open-clip").
        ckpt: Checkpoint or specific model identifier (e.g., "ViT-B-16/openai").
    """

    org: str
    ckpt: str
    _: dataclasses.KW_ONLY
    drop_keys: tuple[str, ...] = dataclasses.field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, dct: dict[str, object]) -> "Model":
        drop_keys = tuple(dct.pop("drop_keys", []))
        return cls(**dct, drop_keys=drop_keys)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Data:
    beluga: str = ""
    """Data pathfor the Beluga whale re-ID benchmark."""
    fishnet: str = ""
    """Data path for the FishNet benchmark."""
    fungiclef: str = ""
    """Data path for the FungiCLEF benchmark."""
    imagenet1k: str = ""
    """Data path for the ImageNet-1K benchmark. You can put anything (like 'huggingface') because it is downloaded from HF."""
    newt: str = ""
    """Data path for the NeWT benchmark."""
    herbarium19: str = ""
    """Data path for the Herbarium19 benchmark."""
    inat21: str = ""
    """Data path for the iNat2021 benchmark."""
    kabr: str = ""
    """Data path for the KABR benchmark."""
    mammalnet: str = ""
    """Data path for the MammalNet benchmark."""
    plantnet: str = ""
    """Data path for the Pl@ntNet benchmark."""
    plankton: str = ""
    """Data path for the planktok classification benchmark."""
    iwildcam: str = ""
    """Data path for the iWildCam benchmark."""

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Experiment:
    """Configuration to run one or more benchmarks in a parallel setting."""

    model: Model

    slurm_acct: str = ""
    """Slurm account. A non-empty string means using Slurm."""
    cfg: str = os.path.join("configs", "neurips.toml")
    """Path to TOML config file."""
    device: typing.Literal["cpu", "mps", "cuda"] = "cuda"
    """which kind of accelerator to use."""
    debug: bool = False
    """whether to run in debug mode."""
    n_train: int = -1
    """Number of maximum training samples. Negative number means use all of them."""
    ssl: bool = True
    """Use SSL when connecting to remote servers to download checkpoints; use --no-ssl if your machine has certificate issues. See `biobench.third_party_models.get_ssl()` for a discussion of how this works."""

    n_workers: int = 4
    """Number of dataloader workers."""
    batch_size: int = 8
    """Initial batch size to start with for tuning."""

    data: Data = dataclasses.field(default_factory=Data)

    report_to: str = os.path.join(".", "results")
    """where to save reports to."""
    log_to: str = os.path.join(".", "logs")
    """where to save logs to."""
    seed: int = 17
    """Random seed."""
    verbose: bool = False
    """DEBUG logging or not."""

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def update(self, other):
        return dataclasses.replace(
            other,
            device=self.device,
            debug=self.debug,
            n_train=self.n_train,
            parallel=self.parallel,
        )


def load(path: str) -> list[Experiment]:
    """Load experiments from a TOML file.

    None of the fields in Experiment are lists, so anytime we find a list in the TOML, we add another dimension to our grid search over all possible experiments.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"TOML file {path} must contain a dictionary at the root level"
        )

    # Extract models list
    models = raw.pop("models", [])
    if not isinstance(models, list):
        raise ValueError("models must be a list of tables in TOML")

    # Start with models as base experiments
    experiments = [{"model": Model.from_dict(model)} for model in models]

    # Handle data config specially
    data = raw.pop("data", {})

    # For each remaining field in the TOML
    for key, value in raw.items():
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
        exp["data"] = Data(**data)

    # Convert dictionaries to Experiment objects
    return [Experiment(**exp) for exp in experiments]
