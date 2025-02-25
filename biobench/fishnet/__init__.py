"""
# FishNet: Fish Recognition, Detection, and Functional Traits Prediction

FishNet ([paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.pdf), [code](https://github.com/faixan-khan/FishNet)) is a large-scale diverse dataset containing 94,532 images from 17,357 aquatic species.
It contains three benchmarks: fish classification, fish detection, and functional traits prediction.

We mainly focus on the third task.
We train an two-layer MLP on the visual features extracted by different model backbones to predict the presence or absence of 9 different traits.

If you use this evaluation, be sure to cite the original work:

```
@InProceedings{Khan_2023_ICCV,
    author    = {Khan, Faizan Farooq and Li, Xiang and Temple, Andrew J. and Elhoseiny, Mohamed},
    title     = {FishNet: A Large-scale Dataset and Benchmark for Fish Recognition, Detection, and Functional Trait Prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20496-20506}
}
```

This task was contributed by [Jianyang Gu](https://vimar-gu.github.io/).
"""

import asyncio
import dataclasses
import logging
import os.path
import random
import typing

import beartype
import numpy as np
import polars as pl
import sklearn
import torch
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

from .. import helpers, interfaces, mllms, registry

logger = logging.getLogger("fishnet")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """FishNet task arguments."""

    data: str = ""
    """dataset directory; where you downloaded this task's data to."""
    batch_size: int = 256
    """batch size for computer vision model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of epochs) to log progress."""
    n_epochs: int = 100
    """How many epochs to train the MLP classifier."""
    learning_rate: float = 5e-4
    """The learning rate for training the MLP classifier."""
    threshold: float = 0.5
    """The threshold to predict "presence" rather than "absence"."""
    seed: int = 42
    """random seed."""

    # Computed at runtime.
    device: str = "cuda"
    """(computed at runtime) which kind of accelerator to use."""
    debug: bool = False
    """(computed at runtime) whether to run in debug mode."""
    n_train: int = -1
    """Number of maximum training samples. Negative number means use all of them."""
    n_test: int = -1
    """Number of test samples. Negative number means use all of them."""
    parallel: int = 1
    """Number of parallel requests per second to MLLM service providers."""


@jaxtyped(typechecker=beartype.beartype)
class FeaturesCvml(torch.utils.data.Dataset):
    """
    A dataset of learned features (dense vectors).
    """

    x: Float[Tensor, " n dim"]
    """Dense feature vectors from a vision backbone."""
    y: Int[Tensor, " n 9"]
    """0/1 labels of absence/presence of 9 different traits."""
    ids: list[str]
    """Image ids."""

    def __init__(
        self,
        x: Float[Tensor, " n dim"],
        y: Int[Tensor, " n n_classes"],
        ids: list[str],
    ):
        self.x = x
        self.y = y
        self.ids = ids

    @property
    def dim(self) -> int:
        """Dimension of the dense feature vectors."""
        _, dim = self.x.shape
        return dim

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, index
    ) -> tuple[Float[Tensor, " dim"], Int[Tensor, " n_classes"], str]:
        return self.x[index], self.y[index], self.ids[index]


@beartype.beartype
def init_classifier(input_dim: int) -> torch.nn.Module:
    """A simple MLP classifier consistent with the design in FishNet."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 512),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 9),
    )


@beartype.beartype
def calc_macro_f1(preds: list[interfaces.Prediction]) -> float:
    """
    Calculate the macro-averaged F1 score across all fish trait predictions.

    For each fish image, we predict 9 binary traits:

    1. Feeding Path (benthic/pelagic)
    2. Tropical habitat (yes/no)
    3. Temperate habitat (yes/no)
    4. Subtropical habitat (yes/no)
    5. Boreal habitat (yes/no)
    6. Polar habitat (yes/no)
    7. Freshwater habitat (yes/no)
    8. Saltwater habitat (yes/no)
    9. Brackish water habitat (yes/no)

    The macro-averaging:

    1. Calculates an F1 score for each trait independently
    2. Takes the unweighted mean of these 9 F1 scores

    This ensures each trait contributes equally to the final score, regardless of class imbalance in the dataset (e.g., if there are many more tropical fish than brackish water fish).

    Args:
        preds: List of predictions, each containing:
            - info["y_pred"]: List of 9 binary predictions
            - info["y_true"]: List of 9 binary ground truth values

    Returns:
        The macro-averaged F1 score across all 9 traits
    """
    y_pred = np.array([pred.info["y_pred"] for pred in preds])
    y_true = np.array([pred.info["y_true"] for pred in preds])
    return sklearn.metrics.f1_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )


@beartype.beartype
def benchmark_cvml(
    args: Args, model_args: interfaces.ModelArgsCvml
) -> tuple[interfaces.ModelArgsCvml, interfaces.TaskReport]:
    """
    The FishNet benchmark.
    """
    # 1. Load model.
    backbone = registry.load_vision_backbone(model_args)

    # 2. Get features.
    train_dataset = get_features(args, backbone, is_train=True)
    test_dataset = get_features(args, backbone, is_train=False)

    # 3. Set up classifier.
    classifier = init_classifier(train_dataset.dim).to(args.device)

    # 4. Load datasets for classifier.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 5. Fit the classifier.
    for epoch in range(args.n_epochs):
        total = 2 if args.debug else len(train_loader)
        it = iter(train_loader)
        for b in range(total):
            features, labels, _ = next(it)
            features = features.to(args.device)
            labels = labels.to(args.device, dtype=torch.float)
            output = classifier(features)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate the classifier.
        if (epoch + 1) % args.log_every == 0:
            examples = evaluate(args, classifier, test_loader)
            score = calc_macro_f1(examples)
            logger.info("Epoch %d/%d: %.3f", epoch + 1, args.n_epochs, score)

    return model_args, interfaces.TaskReport(
        "FishNet", examples, calc_mean_score=calc_macro_f1
    )


@beartype.beartype
def evaluate(
    args: Args, classifier: torch.nn.Module, dataloader
) -> list[interfaces.Prediction]:
    """
    Evaluates the trained classifier on a test split.

    Returns:
        a list of Examples.
    """
    total = 2 if args.debug else len(dataloader)
    it = iter(dataloader)
    examples = []
    for b in range(total):
        features, labels, ids = next(it)
        features = features.to(args.device)
        labels = labels.numpy()
        with torch.no_grad():
            pred_logits = classifier(features)
        pred_logits = (pred_logits > args.threshold).cpu().numpy()
        for id, pred, true in zip(ids, pred_logits, labels):
            example = interfaces.Prediction(
                id,
                float((pred == true).all()),
                {"y_pred": pred.tolist(), "y_true": true.tolist()},
            )
            examples.append(example)

    return examples


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, *, is_train: bool
) -> FeaturesCvml:
    """Extract visual features."""
    if not os.path.isdir(args.data):
        msg = f"Path '{args.data}' doesn't exist. Did you download the FishNet dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--fishnet-args.data'; see --help for more."
        raise ValueError(msg)

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    file = "train.csv" if is_train else "test.csv"
    dataset = ImageDatasetCvml(args.data, file, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=True,
    )

    all_features, all_labels, all_ids = [], [], []

    if args.debug:
        n = args.batch_size * 2
        total = 2  # 2 batches
    elif is_train and args.n_train >= 0:
        n = args.n_train
        total = n // args.batch_size + 1
    elif not is_train and args.n_test > 0:
        n = args.n_test
        total = n // args.batch_size + 1
    else:
        n = len(dataset)
        total = len(dataloader)

    it = iter(dataloader)
    for b in helpers.progress(range(total), every=args.log_every, desc=file):
        images, labels, ids = next(it)
        images = images.to(args.device)

        features = backbone.img_encode(images).img_features
        all_features.append(features.cpu())
        all_labels.append(labels)

        all_ids.extend(ids)

    # Keep the Tensor data type for subsequent training
    all_features = torch.cat(all_features, dim=0)[:n]
    all_labels = torch.cat(all_labels, dim=0)[:n]
    all_ids = all_ids[:n]

    return FeaturesCvml(all_features, all_labels, all_ids)


@jaxtyped(typechecker=beartype.beartype)
class ImageDatasetCvml(torch.utils.data.Dataset):
    """
    A dataset for CV+ML that loads the required attribute labels.
    """

    def __init__(self, root_dir: str, csv_file: str, transform):
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir, csv_file)
        self.df = pl.read_csv(self.csv_file).with_row_index()
        self.all_columns = [
            "FeedingPath",
            "Tropical",
            "Temperate",
            "Subtropical",
            "Boreal",
            "Polar",
            "freshwater",
            "saltwater",
            "brackish",
        ]
        for col in self.all_columns:
            self.df = self.df.filter(self.df[col].is_not_null())
        self.transform = transform

        # Corresponding column indices
        self.image_col = 4
        self.folder_col = 13
        self.label_cols = [15, 16, 17, 18, 19, 20, 21, 22, 23]
        logger.info("csv file: %s has %d item.", csv_file, len(self.df))

    def __getitem__(
        self, index: int
    ) -> tuple[Float[Tensor, "3 width height"], Int[Tensor, "9"], str]:
        row_data = self.df.row(index)
        image_name = row_data[self.image_col]
        image_name = image_name.split("/")[-1]
        folder = row_data[self.folder_col]
        image_path = os.path.join(self.root_dir, "Image_Library", folder, image_name)
        image = Image.open(image_path)

        # Extract the required attribute labels.
        label = []
        for col in self.label_cols:
            value = row_data[col]
            if col == 15:
                if value == "pelagic":
                    value = 1
                elif value == "benthic":
                    value = 0
                else:
                    raise ValueError("FeedingPath can only be pelagic or benthic.")
            label.append(value)
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path

    def __len__(self) -> int:
        return len(self.df)


def benchmark_mllm(
    args: Args, model_args: interfaces.ModelArgsMllm
) -> tuple[interfaces.ModelArgsMllm, interfaces.TaskReport]:
    if not os.path.isdir(args.data):
        msg = f"Path '{args.data}' doesn't exist. Did you download the FishNet dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--fishnet-args.data'; see --help for more."
        raise ValueError(msg)

    train_dataset = DatasetMllm(args.data, "train.csv")
    test_dataset = DatasetMllm(args.data, "test.csv")
    rng = random.Random(args.seed)

    with asyncio.Runner() as loop:
        limiter = mllms.RateLimiter(args.parallel)
        semaphore = asyncio.Semaphore(args.parallel)

        # We load all the training samples into memory right away because they will be re-used over and over again.
        # Test samples are loaded one by one on demand.
        i_train = rng.sample(range(len(train_dataset)), k=args.n_train)
        train_examples = [
            train_dataset[i].to_example()
            for i in helpers.progress(i_train, desc="load train samples")
        ]

        @beartype.beartype
        async def run_one(i: int) -> interfaces.Prediction:
            async with semaphore:
                test_example = test_dataset[i]
                # Try to fit them into a prompt.
                n_examples = 0
                fewshot_examples = []
                while mllms.fits(
                    model_args,
                    fewshot_examples,
                    test_example.image_b64,
                    test_example.user,
                ) and (args.n_train < 0 or n_examples < args.n_train):
                    # Add another example.
                    n_examples += 1
                    fewshot_examples = train_examples[:n_examples]

                # Only shuffle once.
                rng.shuffle(fewshot_examples)

                await limiter.acquire()
                assistant = await mllms.send(
                    model_args,
                    fewshot_examples,
                    test_example.image_b64,
                    test_example.user,
                )
                preds = test_example.parse_assistant(assistant)
                # Convert feeding path to bool and combine with other binary values
                preds = [preds[1] == "pelagic"] + list(preds[2:])

                return interfaces.Prediction(
                    test_example.image_id,
                    float(preds == test_example.true),
                    info={
                        "y_true": test_example.true,
                        "y_pred": preds,
                        "assistant": assistant,
                    },
                )

        @beartype.beartype
        async def run_all() -> list[interfaces.Prediction]:
            if args.debug:
                logger.info("Using the first 10 samples out of %d.", len(test_dataset))
                test_i = list(range(10))
            elif args.n_test >= 0:
                logger.info(
                    "Using %d random samples out of %d.", args.n_test, len(test_dataset)
                )
                test_i = rng.sample(range(len(test_dataset)), k=args.n_test)

            jobs = [asyncio.create_task(run_one(i)) for i in test_i]

            preds = []
            for job in helpers.progress(jobs):
                pred: interfaces.Prediction = await job
                preds.append(pred)
            return preds

        preds = loop.run(run_all())

    return model_args, interfaces.TaskReport(
        "FishNet", args.n_train, preds, calc_mean_score=calc_macro_f1
    )


@dataclasses.dataclass(frozen=True)
class SampleMllm:
    image_id: str
    image_b64: str
    trophic_level: float
    """The position in the food chain or food web."""
    feeding_path: typing.Literal["benthic", "pelagic"]
    """Which trophic pathway the species feeds from."""
    tropical: bool
    """Whether the fish can live in tropical areas."""
    temperate: bool
    """Whether the fish can live in temperate areas."""
    subtropical: bool
    """Whether the fish can live in subtropical areas."""
    boreal: bool
    """Whether the fish can live in boral areas."""
    polar: bool
    """Whether the fish can live in polar areas."""
    freshwater: bool
    """Whether the fish can live in freshwater."""
    saltwater: bool
    """Whether the fish can live in saltwater."""
    brackish: bool
    """Whether the fish can live in brackish water."""

    @property
    def user(self) -> str:
        return """
What functional traits does this fish have? For each of these ten traits, respond using the specified format:
* **trophic level**: What is the real-valued position of this fish in the food chain or food web?
* **feeding path**: What trophic pathway does this fish feed from? Choose between benthic or pelagic.
* **tropical**: Can this fish live in tropical areas? Respond with yes or no.
* **temperate**: Can this fish live in temperate areas? Respond with yes or no.
* **subtropical**: Can this fish live in subtropical areas? Respond with yes or no.
* **boreal**: Can this fish live in boreal areas? Respond with yes or no.
* **polar**: Can this fish live in polar areas? Respond with yes or no.
* **freshwater**: Can this fish live in freshwater? Respond with yes or no.
* **saltwater**: Can this fish live in saltwater? Respond with yes or no.
* **brackish water**: Can this fish live in brackish water? Respond with yes or no.
""".strip()

    @property
    def assistant(self) -> str:
        return f"""
* **trophic level**: {self.trophic_level:.2f}
* **feeding path**: {self.feeding_path}
* **tropical**: {"yes" if self.tropical else "no"}
* **subtropical**: {"yes" if self.subtropical else "no"}
* **subtropical**: {"yes" if self.subtropical else "no"}
* **boreal**: {"yes" if self.boreal else "no"}
* **polar**: {"yes" if self.polar else "no"}
* **freshwater**: {"yes" if self.freshwater else "no"}
* **saltwater**: {"yes" if self.saltwater else "no"}
* **brackish water**: {"yes" if self.brackish else "no"}""".strip()

    def parse_assistant(
        self, assistant: str
    ) -> tuple[float, str, bool, bool, bool, bool, bool, bool, bool, bool]:
        """Parse the MLLM response into structured data.

        Returns:
            Tuple of (trophic_level, feeding_path, tropical, temperate, subtropical, boreal, polar, freshwater, saltwater, brackish)
        """
        # Default values
        trophic_level = 0.0
        feeding_path = "benthic"  # Default to benthic if not found
        tropical = temperate = subtropical = boreal = polar = False
        freshwater = saltwater = brackish = False

        # Parse each line
        for line in assistant.split("\n"):
            line = line.strip()
            if not line.startswith("*"):
                continue

            # Extract the trait name and value
            parts = line.split(":")
            if len(parts) != 2:
                continue

            trait = parts[0].strip("* ").lower()
            value = parts[1].strip().lower()

            if trait == "trophic level":
                try:
                    trophic_level = float(value)
                except ValueError:
                    pass
            elif trait == "feeding path":
                if value in ("benthic", "pelagic"):
                    feeding_path = value
            elif trait == "tropical":
                tropical = value == "yes"
            elif trait == "temperate":
                temperate = value == "yes"
            elif trait == "subtropical":
                subtropical = value == "yes"
            elif trait == "boreal":
                boreal = value == "yes"
            elif trait == "polar":
                polar = value == "yes"
            elif trait == "freshwater":
                freshwater = value == "yes"
            elif trait == "saltwater":
                saltwater = value == "yes"
            elif trait == "brackish water":
                brackish = value == "yes"

        return (
            trophic_level,
            feeding_path,
            tropical,
            temperate,
            subtropical,
            boreal,
            polar,
            freshwater,
            saltwater,
            brackish,
        )

    @property
    def true(self) -> list[bool]:
        """Get the ground truth binary values for all traits except trophic level.

        Returns:
            List of boolean values for [feeding_path=='pelagic', tropical, temperate,
            subtropical, boreal, polar, freshwater, saltwater, brackish]
        """
        return [
            self.feeding_path == "pelagic",
            self.tropical,
            self.temperate,
            self.subtropical,
            self.boreal,
            self.polar,
            self.freshwater,
            self.saltwater,
            self.brackish,
        ]

    def to_example(self) -> mllms.Example:
        return mllms.Example(
            image_b64=self.image_b64,
            user=self.user,
            assistant=self.assistant,
        )


@jaxtyped(typechecker=beartype.beartype)
class DatasetMllm(torch.utils.data.Dataset):
    """
    A dataset for MLLMs that loads the required attribute labels.
    """

    def __init__(self, root_dir: str, csv_file: str):
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir, csv_file)
        self.df = pl.read_csv(self.csv_file).with_row_index()
        self.all_columns = [
            "FeedingPath",
            "Tropical",
            "Temperate",
            "Subtropical",
            "Boreal",
            "Polar",
            "freshwater",
            "saltwater",
            "brackish",
        ]
        for col in self.all_columns:
            self.df = self.df.filter(self.df[col].is_not_null())

        # Corresponding column indices
        self.img_col = 4
        self.folder_col = 13
        logger.info("csv file: %s has %d items.", csv_file, len(self.df))

    def __getitem__(self, index: int) -> SampleMllm:
        row_data = self.df.row(index)
        img_name = row_data[self.img_col]
        img_name = img_name.split("/")[-1]
        folder = row_data[self.folder_col]
        img_path = os.path.join(self.root_dir, "Image_Library", folder, img_name)
        img_b64 = helpers.load_img_b64(img_path)

        return SampleMllm(
            img_path,
            img_b64,
            row_data[14],
            row_data[15],
            bool(row_data[16]),
            bool(row_data[17]),
            bool(row_data[18]),
            bool(row_data[19]),
            bool(row_data[20]),
            bool(row_data[21]),
            bool(row_data[22]),
            bool(row_data[23]),
        )

    def __len__(self) -> int:
        return len(self.df)
