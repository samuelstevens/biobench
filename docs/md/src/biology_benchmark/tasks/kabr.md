Module src.biology_benchmark.tasks.kabr
=======================================
# Kenyan Animal Behavior Recognition (KABR)

KABR is a video recognition task ([paper](https://openaccess.thecvf.com/content/WACV2024W/CV4Smalls/papers/Kholiavchenko_KABR_In-Situ_Dataset_for_Kenyan_Animal_Behavior_Recognition_From_Drone_WACVW_2024_paper.pdf), [website](https://kabrdata.xyz/), [Huggingface](https://huggingface.co/datasets/imageomics/KABR)) where the model predicts Kenyan animal behavior in short video segments.

This can be framed as a classification task: given a short video segment of a single animal, which behavior is most common within the segment?

While specialized architectures exist, we train a simple nearest-centroid classifier [which works well with few-shot tasks](https://arxiv.org/abs/1911.04623) over video representations.
We get video representations by embedding each frame of the video and taking the mean over the batch dimension.

## Data

To download the data, you need to use the dataset download script:

1. Copy-paste the [download script](https://huggingface.co/datasets/imageomics/KABR/raw/main/download.py) to your data directory, like `/local/scratch/datasets/KABR/download.py`.
2. Run `python download.py`. It doesn't have any requirements beyond the python standard library.

Functions
---------

`aggregate_frames(args: src.biology_benchmark.tasks.kabr.Args, features: jaxtyping.Float[Tensor, 'batch']) ‑> jaxtyping.Float[Tensor, 'batch']`
:   

`aggregate_labels(args: src.biology_benchmark.tasks.kabr.Args, labels: jaxtyping.Float[Tensor, 'batch']) ‑> jaxtyping.Float[Tensor, 'batch']`
:   

`batched_idx(total_size: int, batch_size: int) ‑> collections.abc.Iterator[tuple[int, int]]`
:   

`get_features(args: src.biology_benchmark.tasks.kabr.Args, model: biology_benchmark.models.interfaces.VisionModel, dataloader) ‑> tuple[jaxtyping.Float[Tensor, 'batch'], jaxtyping.Float[Tensor, 'batch']]`
:   

`l2_normalize(features: jaxtyping.Float[ndarray, 'batch']) ‑> jaxtyping.Float[Tensor, 'batch']`
:   

`main(args: src.biology_benchmark.tasks.kabr.Args)`
:   

`simpleshot(args: src.biology_benchmark.tasks.kabr.Args, x_features: jaxtyping.Float[ndarray, 'batch'], x_labels, y_features, y_labels) ‑> jaxtyping.Float[ndarray, '']`
:   Applies simpleshot to the video clips. We assign each clip the majority label.

Classes
-------

`Args(seed: int = 42, model: biology_benchmark.models.Params = <factory>, dataset_dir: str = '', batch_size: int = 2048, n_workers: int = 4, device: Literal['cpu', 'cuda'] = 'cpu')`
:   Args(seed: int = 42, model: biology_benchmark.models.Params = <factory>, dataset_dir: str = '', batch_size: int = 2048, n_workers: int = 4, device: Literal['cpu', 'cuda'] = 'cpu')

    ### Class variables

    `batch_size: int`
    :   batch size for linear model.

    `dataset_dir: str`
    :   dataset directory; where you downloaded KABR to.

    `device: Literal['cpu', 'cuda']`
    :

    `model: biology_benchmark.models.Params`
    :

    `n_workers: int`
    :   number of dataloader worker processes.

    `seed: int`
    :   random seed.

`Kabr(path, split: str, transform=None, seed: int = 42)`
:   Clips of at most 90 frames in Charades format with each frame stored as an image.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

`Video(video_id: str, frames: list[str], labels: list[int])`
:   Video(video_id: str, frames: list[str], labels: list[int])

    ### Class variables

    `frames: list[str]`
    :

    `labels: list[int]`
    :

    `video_id: str`
    :