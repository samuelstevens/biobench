# Adding New Models

To add a new model, you need to:

1. Write a new class that satisfies `biobench.interfaces.VisionBackbone`.
2. Register the class using `register_vision_backbone` under a new name.

## 1. Create a New Class

Here is an example of writing a new class for [pre-trained torchvision models](https://pytorch.org/vision/stable/models.html).
First, create a new class that will satisy the interface:

```python
@jaxtyped(typechecker=beartype.beartype)
class TorchvisionModel(biobench.interfaces.VisionBackbone):
    def __init__(self, ckpt: str):
        import torchvision

        arch, weights = ckpt.split("/")
        if not hasattr(torchvision, arch):
            raise ValueError(f"'{arch}' is not a valid torchvision architecture.")
        self.model = getattr(torchvision, arch)(weights=weights)
        self.model.eval()

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> biobench.interfaces.EncodedImgBatch:
        breakpoint()

    def make_img_transform(self):
        # Per the docs, each set of weights has its own transform: https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models
        return self.model.weights.transforms()
```

All models are initialized with a single checkpoint string.
This string can be in whatever format you want, so it is best to document it somewhere so users know how to format it.

Here, we will split on `/` and then assume the first half is a valid classname on `torchvision`.
To make this a little easier on our users, we raise an error that has a more helpful description.
Then we load the weights.

.. tip:: See that we don't import `torchvision` until we finally initialize an instance of `TorchvisionModel`; this means that we don't have the runtime cost of importing `torchvision` if no instance of `TorchvisionModel` is actually used.


## 2. Register the Class

Somewhere in your main script, you need to run:

```py
biobench.registry.register_vision_backbone("torchvision", TorchvisionModel)
```

You can add this line to your copy of `benchmark` to use `--model torchvision resnet50/IMAGENET1K_V2` to load a ResNet50 trained on ImageNet-1K.
