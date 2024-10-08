"""
# FishNet: Fish Recognition, Detection, and Functional Traits Prediction

FishNet is a large-scale diverse dataset containing 94,532 images from 17,357 aquatic species.
It contains three benchmarks: fish classification, fish detection, and functional traits prediction.

We mainly focus on the third task.
We train a classifier on the visual features extracted by different model backbones, and evaluate on test data.

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
"""
import tqdm
import torch
import os.path
import sklearn
import logging
import beartype
import dataclasses
import numpy as np
import polars as pl
from PIL import Image
from torch import Tensor
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped

from biobench import interfaces, registry

logger = logging.getLogger("fishnet")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    """FishNet task arguments."""

    batch_size: int = 256
    """batch size for deep model."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of batches) to log progress."""
    n_epochs: int = 100
    """how many epochs to fit the linear classifier."""
    learning_rate: float = 5e-4
    """the learning rate for fine-tuning the classifier."""
    threshold: float = 0.5
    """the threshold to transfer predicted logits."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float[Tensor, " n dim"]
    y: Int[Tensor, " n n_classes"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
class FeatureDataset(torch.utils.data.Dataset):
    """The dataset for the Features class"""
    def __init__(self, features: Features):
        self.x = features.x
        self.y = features.y
        self.ids = features.ids
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.ids[index])


@beartype.beartype
class Classifier(torch.nn.Module):
    """A simple MLP classifier consistent with the design in FishNet."""
    def __init__(self, final_layer_dim: int):
        super(Classifier, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(final_layer_dim, 512),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 9)
        )
    
    def forward(self, x):
        """"""
        return self.linear(x)


@beartype.beartype
class MeanScoreCalculator:
    def __call__(self, examples: list[interfaces.Example]) -> float:
        y_pred = np.array([example.info["y_pred"] for example in examples])
        y_true = np.array([example.info["y_true"] for example in examples])
        score = sklearn.metrics.f1_score(
            y_true, y_pred, average="macro", labels=np.unique(y_true)
        )
        return score.item()


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgs
) -> tuple[interfaces.ModelArgs, interfaces.TaskReport]:
    """
    The FishNet benchmark.
    """
    # 1. Load model.
    backbone = registry.load_vision_backbone(*model_args)
    for dim_index in range(3):
        try:
            if dim_index == 0:
                embed_dim = backbone.model.output_dim
            elif dim_index == 1:
                embed_dim = backbone.model.embed_dim
            elif dim_index == 2:
                embed_dim = backbone.model.trunk.embed_dim
            break
        except AttributeError:
            if dim_index < 3:
                continue
    assert embed_dim is not None, "Backbone embedding dimension type not defined."
    classifier = Classifier(embed_dim).to(args.device)

    # 2. Get features.
    train_features, test_features = get_features(args, backbone)

    # 3. Load datasets for classifier.
    train_dataset = FeatureDataset(train_features)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, pin_memory=True
    )
    test_dataset = FeatureDataset(test_features)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=False
    )
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.learning_rate
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    true_labels = test_features.y

    # 4. Fit the classifier.
    best_score = 0.
    for epoch in range(args.n_epochs):
        total = len(train_loader)
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
            total_test = len(test_loader)
            it_test = iter(test_loader)
            examples = []
            for b_test in range(total_test):
                features_test, labels_test, ids_test = next(it_test)
                features_test = features_test.to(args.device)
                labels_test = labels_test.numpy()
                ids_test = ids_test.numpy()
                with torch.no_grad():
                    pred_logits = classifier(features_test)
                pred_logits = (pred_logits > args.threshold).cpu().numpy()
                for image_id, pred, true in zip(
                    ids_test, pred_logits, labels_test
                ):
                    examples.append(interfaces.Example(
                        str(image_id),
                        float((pred == true).all()),
                        {"y_pred": pred.tolist(), "y_true": true.tolist()},
                    ))
                score = MeanScoreCalculator()(examples)
            logger.info(f"{epoch + 1}/{args.n_epochs}: {score}")
    
    return model_args, interfaces.TaskReport(
        'FishNet', examples, MeanScoreCalculator()
    )


@jaxtyped(typechecker=beartype.beartype)
def get_features(args, backbone) -> tuple[Features, Features]:
    """Get the features with the specified backbone."""
    transform = backbone.make_img_transform()
    train_dataset = ImageDataset(args.datadir, 'train.csv',
                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, pin_memory=False
    )
    train_features = get_split_features(args, backbone, train_loader)

    test_dataset = ImageDataset(args.datadir, 'test.csv',
                                transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=False
    )
    test_features = get_split_features(args, backbone, test_loader)

    return train_features, test_features


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_split_features(
    args: Args, backbone: interfaces.VisionBackbone, dataloader
) -> Features:
    """Extract visual features"""
    backbone = torch.compile(backbone.to(args.device))

    all_features, all_labels, all_ids = [], [], []

    total = len(dataloader)
    logger.info("Extracting features.")
    it = iter(dataloader)
    for b in tqdm.tqdm(range(total)):
        images, labels, _ = next(it)
        images = images.to(args.device)

        features = backbone.img_encode(images).img_features
        all_features.append(features.cpu())
        all_labels.append(labels)

        ids = (np.arange(len(labels)) + b * args.batch_size)
        all_ids.append(ids)

    # Keep the Tensor data type for subsequent training
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_ids = np.concatenate(all_ids, axis=0)

    return Features(all_features, all_labels, all_ids)


@jaxtyped(typechecker=beartype.beartype)
class ImageDataset(torch.utils.data.Dataset):
    """
    A dataset that loads the required attribute labels.
    """

    def __init__(self, root_dir: str, csv_file: str, transform):
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir, csv_file)
        self.df = pl.read_csv(self.csv_file).with_row_index()
        self.all_columns = [
            "FeedingPath", "Tropical", "Temperate", "Subtropical", "Boreal",
            "Polar", "freshwater", "saltwater", "brackish"
        ]
        for col in self.all_columns:
            self.df = self.df.filter(self.df[col].is_not_null())
        self.transform = transform

        # Corresponding column indices
        self.image_col = 4
        self.folder_col = 13
        self.label_cols = [15, 16, 17, 18, 19, 20, 21, 22, 23]
        logger.info('csv file: {} has {} item.'.format(csv_file, len(self.df)))

    def __getitem__(
        self, index: int
    ) -> tuple[Float[Tensor, "3 width height"], Float[Tensor, "channel"], str]:
        row_data = self.df.row(index)
        image_name = row_data[self.image_col]
        image_name = image_name.split('/')[-1]
        folder = row_data[self.folder_col]
        image_path = os.path.join(
            self.root_dir, 'Image_Library', folder, image_name
        )
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
                    raise ValueError(
                        "FeedingPath can only be pelagic or benthic."
                    )
            label.append(value)
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)
        
        return image, label, image_path

    def __len__(self) -> int:
        return len(self.df)
