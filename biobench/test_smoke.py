def test_smoke_imports():
    """Test that all modules can be imported without errors."""
    import biobench
    import biobench.aimv2
    import biobench.config
    import biobench.fishnet
    import biobench.fungiclef
    import biobench.fungiclef.metrics
    import biobench.helpers
    import biobench.herbarium19
    import biobench.inat21
    import biobench.leopard
    import biobench.openset
    import biobench.rarespecies
    import biobench.registry
    import biobench.reporting
    import biobench.simpleshot
    import biobench.third_party_models
    import biobench.vjepa

    # Check that key classes/functions exist
    assert hasattr(biobench.registry, "VisionBackbone")
    assert hasattr(biobench.config, "Experiment")
    assert hasattr(biobench.fungiclef, "benchmark")


def test_basic_instantiation():
    """Test that basic objects can be instantiated."""
    from biobench.config import Data, Experiment, Model

    # Create minimal valid objects
    model = Model(org="test", ckpt="test")
    data = Data()
    experiment = Experiment(model=model, data=data, seed=42)

    # Verify they have expected properties
    assert experiment.model.org == "test"
    assert experiment.seed == 42
