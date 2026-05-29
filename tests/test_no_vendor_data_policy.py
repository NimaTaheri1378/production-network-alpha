from pathlib import Path


def test_gitignore_blocks_raw_vendor_data():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text()
    assert "data/raw/*" in gitignore
    assert "artifacts/model_runs/*" in gitignore
    assert ".pgpass" in gitignore
