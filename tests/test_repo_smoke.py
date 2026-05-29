from pathlib import Path


def test_package_imports():
    import production_network_alpha

    assert production_network_alpha.__version__


def test_core_configs_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "configs/data.yml",
        "configs/features.yml",
        "configs/backtest.yml",
        "configs/schema_map.yml",
    ]:
        assert (root / rel).exists()
