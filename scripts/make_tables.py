def make_overview_table():
    """Make a table for the entire combination of tasks and models."""
    models = {
        "ViT-L-14-384/openai": "CLIP ViT-L/14 (384px)",
        "ViT-L-14/openai": "CLIP ViT-L/14",
        "ViT-B-16/openai": "CLIP ViT-B/16",
        "ViT-SO400M-14-SigLIP-384/webli": "SigLIP ViT-SO400M/14 (384px)",
        "ViT-SO400M-14-SigLIP/webli": "SigLIP ViT-SO400M/14",
        "ViT-L-16-SigLIP-384/webli": "SigLIP ViT-L/16 (384px)",
        "ViT-L-16-SigLIP-256/webli": "SigLIP ViT-L/16 (256px)",
        "ViT-B-16-SigLIP-512/webli": "SigLIP ViT-B/16 (512px)",
        "ViT-B-16-SigLIP-384/webli": "SigLIP ViT-B/16 (384px)",
        "ViT-B-16-SigLIP-256/webli": "SigLIP ViT-B/16 (256px)",
        "ViT-B-16-SigLIP/webli": "SigLIP ViT-B/16",
        "vit_giant_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-g/14",
        "vit_large_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-L/14",
        "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-B/14",
        "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-S/14",
        "hf-hub:imageomics/bioclip": "BioCLIP ViT-B/16",
    }

    tasks = {
        "newt": "NeWT",
        "imagenet1k": "ImageNet-1K",
        "plankton": "Plankton",
        "fishnet": "FishNet",
        "herbarium19": "Herbarium19",
        "kabr": "KABR",
        "iwildcam": "iWildCam",
        "plantnet": "Pl@ntNet",
        "mammalnet": "MammalNet",
    }
