import json

import beartype
import mlcroissant as mlc
import tyro


@beartype.beartype
def write_croissant(path: str = "croissant.json"):
    """Create Croissant metadata for WildVision benchmark with download scripts"""

    metadata = mlc.Metadata(
        name="BioBench",
        description="A benchmark suite of 9 application-driven computer vision tasks in ecology spanning 4 taxonomic kingdoms",
        license="MIT",
        url="https://samuelstevens.me/biobench",
        cite_as="@software{stevens2025biobench,author={Stevens, Samuel and Gu, Jianyang},license = {MIT},title = {{BioBench}},url = {https://github.com/samuelstevens/biobench/}}",
        version="1.0",
    )

    print(metadata.issues.report())

    with open(path, "w") as fd:
        content = metadata.to_json()
        fd.write(json.dumps(content, indent=4) + "\n")


@beartype.beartype
def load_croissant(path: str = "croissant.json"):
    dataset = mlc.Dataset(path)
    breakpoint()


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "write": write_croissant,
        "load": load_croissant,
    })
