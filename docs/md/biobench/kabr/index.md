Module biobench.kabr
====================
# Kenyan Animal Behavior Recognition (KABR)

KABR is a video recognition task ([paper](https://openaccess.thecvf.com/content/WACV2024W/CV4Smalls/papers/Kholiavchenko_KABR_In-Situ_Dataset_for_Kenyan_Animal_Behavior_Recognition_From_Drone_WACVW_2024_paper.pdf), [website](https://kabrdata.xyz/), [Huggingface](https://huggingface.co/datasets/imageomics/KABR)) where the model predicts Kenyan animal behavior in short video segments.

This can be framed as a classification task: given a short video segment of a single animal, which behavior is most common within the segment?

While specialized architectures exist, we train a simple nearest-centroid classifier [which works well with few-shot tasks](https://arxiv.org/abs/1911.04623) over video representations.
We get video representations by embedding each frame of the video and taking the mean over the batch dimension.

## Data

To download the data, you need to use the dataset download script:

1. Copy-paste the [download script](https://huggingface.co/datasets/imageomics/KABR/raw/main/download.py) to your data directory, like `/scratch/KABR/download.py`.
2. Run `python download.py`. It doesn't have any requirements beyond the Python standard library.

Functions
---------

`aggregate_frames(args: biobench.kabr.Args, features: jaxtyping.Float[Tensor, 'n_frames n_examples dim']) ‑> jaxtyping.Float[Tensor, 'n_examples dim']`
:   

`aggregate_labels(args: biobench.kabr.Args, labels: jaxtyping.Int[Tensor, 'n_frames n_examples']) ‑> jaxtyping.Int[Tensor, 'n_examples']`
:   

`batched_idx(total_size: int, batch_size: int) ‑> collections.abc.Iterator[tuple[int, int]]`
:   

`benchmark(backbone: biobench.interfaces.VisionBackbone, args: biobench.kabr.Args) ‑> biobench.interfaces.BenchmarkReport`
:   

`get_features(args: biobench.kabr.Args, backbone: biobench.interfaces.VisionBackbone, dataloader) ‑> tuple[jaxtyping.Float[Tensor, 'n_frames n_examples dim'], jaxtyping.Int[Tensor, 'n_frames n_examples']]`
:   Gets all model features and true labels for all frames and all examples in the dataloader.
    
    Returns it as a pair of big Tensors.

`l2_normalize(features: jaxtyping.Float[Tensor, 'n_examples dim']) ‑> jaxtyping.Float[Tensor, 'n_examples dim']`
:   

`simpleshot(args: biobench.kabr.Args, x_features: jaxtyping.Float[Tensor, 'n_x_examples dim'], x_labels: jaxtyping.Int[Tensor, 'n_x_examples'], y_features: jaxtyping.Float[Tensor, 'n_y_examples dim'], y_labels: jaxtyping.Int[Tensor, 'n_y_examples']) ‑> float`
:   Applies simpleshot to the video clips. We assign each clip the majority label.

Classes
-------

`Args(seed: int = 42, model: biobench.models.Params = <factory>, dataset_dir: str = '', batch_size: int = 16, n_workers: int = 4, frame_agg: Literal['mean', 'max'] = 'mean', device: Literal['cpu', 'cuda'] = 'cuda')`
:   Args(seed: int = 42, model: biobench.models.Params = <factory>, dataset_dir: str = '', batch_size: int = 16, n_workers: int = 4, frame_agg: Literal['mean', 'max'] = 'mean', device: Literal['cpu', 'cuda'] = 'cuda')

    ### Class variables

    `batch_size: int`
    :   batch size for deep model. Note that this is multiplied by 16 (number of frames)

    `dataset_dir: str`
    :   dataset directory; where you downloaded KABR to.

    `device: Literal['cpu', 'cuda']`
    :   which device to use.

    `frame_agg: Literal['mean', 'max']`
    :   how to aggregate features across time dimension.

    `model: biobench.models.Params`
    :

    `n_workers: int`
    :   number of dataloader worker processes.

    `seed: int`
    :   random seed.

`Dataset(path, split: str, transform=None, seed: int = 42)`
:   Clips of at most 90 frames in Charades format with each frame stored as an image.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

`Video(video_id: int, frames: list[str], labels: list[int])`
:   Video(video_id: int, frames: list[str], labels: list[int])

    ### Class variables

    `frames: list[str]`
    :

    `labels: list[int]`
    :

    `video_id: int`
    :