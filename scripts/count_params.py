import fnmatch

import beartype
import tyro

from biobench import config, registry


@beartype.beartype
def main(cfg: str, ckpt: str) -> None:
    for cfg in config.load(cfg):
        if not fnmatch.fnmatch(cfg.model.ckpt, ckpt):
            continue
        model = registry.load_vision_backbone(cfg.model)
        print(
            f"{cfg.model.ckpt} : {sum(params.numel() for params in model.parameters()):_}"
        )


if __name__ == "__main__":
    tyro.cli(main)
