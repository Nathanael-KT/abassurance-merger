import pytest

import src.ml.lake_dataset as ld
import src.ml.train_model as tm
from src.spark.generate_lake import generate_lake


@pytest.fixture()
def isolated_lake(tmp_path, monkeypatch):
    """Data lake de test isole (n'ecrase pas data/hdfs_sim) avec assez de volume."""
    generate_lake(n_clients=3000, seed=42, lake_dir=tmp_path)
    monkeypatch.setattr(tm, "load_lake_tables", lambda: ld.load_lake_tables(tmp_path))
    monkeypatch.setattr(tm, "MODEL_DIR", tmp_path / "artifacts")
    return tmp_path


def test_train_model_returns_reasonable_auc(isolated_lake):
    """US 5.2 -- critere d'acceptation : le modele, entraine sur le data lake sans fuite de
    donnees, est meilleur que le hasard (AUC credible, pas 'trop beau pour etre vrai')."""
    metrics = tm.train()
    assert metrics["source"] == "data_lake_parquet"
    assert metrics["roc_auc"] > 0.65


def test_train_model_is_reproducible(isolated_lake):
    """US 5.1 -- critere d'acceptation : le split train/test et l'entrainement sont
    documentes et reproductibles (seed fixe)."""
    metrics_1 = tm.train()
    metrics_2 = tm.train()
    assert metrics_1["n_train"] == metrics_2["n_train"]
    assert metrics_1["n_test"] == metrics_2["n_test"]
    assert abs(metrics_1["roc_auc"] - metrics_2["roc_auc"]) < 1e-6


def test_train_falls_back_to_synthetic_when_lake_missing(tmp_path, monkeypatch):
    """Resilience : sans data lake, l'entrainement ne plante pas et le signale."""
    monkeypatch.setattr(tm, "load_lake_tables", lambda: ld.load_lake_tables(tmp_path))
    monkeypatch.setattr(tm, "MODEL_DIR", tmp_path / "artifacts")
    assert tm.train()["source"] == "synthetique"
