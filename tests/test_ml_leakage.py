"""Tests IA : data leakage, déséquilibre des classes, validation des features, dérive."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.ml.lake_dataset import (
    DataLeakageError,
    FEATURE_COLUMNS,
    assert_no_leakage,
    build_training_frame,
    class_balance,
    validate_features,
)
from src.ml.train_model import population_drift
from src.spark.generate_lake import generate_lake


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    d = tmp_path_factory.mktemp("lake")
    generate_lake(n_clients=800, seed=7, lake_dir=d)
    return {n: pd.read_parquet(d / f"{n}.parquet")
            for n in ("clients", "contrats", "sinistres", "paiements")}


def test_assert_no_leakage_rejects_contract_end_date():
    """Une variable comme date_fin n'est pas connue au moment de prédire."""
    with pytest.raises(DataLeakageError):
        assert_no_leakage(FEATURE_COLUMNS + ["date_fin"])


def test_features_never_contain_leaky_columns():
    assert_no_leakage(FEATURE_COLUMNS)


def test_features_ignore_events_after_observation_date(lake):
    """Anti-fuite temporelle : ajouter des sinistres APRES la date d'observation ne doit pas
    modifier les features (seule la cible peut changer)."""
    base = build_training_frame(**lake, observation_date="2024-01-01")
    futur = lake["sinistres"].copy()
    extra = futur.head(50).copy()
    extra["sinistre_id"] = extra["sinistre_id"] + 10**6
    extra["date_sinistre"] = "2024-06-15"
    extra["montant_estime"] = 99999.0
    lake2 = dict(lake, sinistres=pd.concat([futur, extra], ignore_index=True))
    after = build_training_frame(**lake2, observation_date="2024-01-01")
    pd.testing.assert_frame_equal(base[FEATURE_COLUMNS], after[FEATURE_COLUMNS])


def test_bug_ia_leakage_inflates_score(lake):
    """BUG IA (fuite de donnees) : si on injecte une variable qui contient la reponse, l'AUC
    devient quasi parfaite. Ce test documente le symptome : un score 'trop beau' est un signal
    d'alerte, et le garde-fou assert_no_leakage doit bloquer cette variable."""
    df = build_training_frame(**lake, observation_date="2024-01-01")
    leaky = df.assign(fuite=df["risque_client"] + np.random.default_rng(0).normal(0, 0.05, len(df)))
    X, y = leaky[FEATURE_COLUMNS + ["fuite"]], leaky["risque_client"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    auc_leak = roc_auc_score(yte, RandomForestClassifier(100, random_state=0)
                             .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    Xc = df[FEATURE_COLUMNS]
    Xtr, Xte, ytr, yte = train_test_split(Xc, y, test_size=0.3, random_state=0, stratify=y)
    auc_clean = roc_auc_score(yte, RandomForestClassifier(100, random_state=0)
                              .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    assert auc_leak > 0.95 > auc_clean
    with pytest.raises(DataLeakageError):
        assert_no_leakage(["fuite", "risque_client"])


def test_class_imbalance_is_profiled(lake):
    df = build_training_frame(**lake, observation_date="2024-01-01")
    balance = class_balance(df["risque_client"])
    assert 0 < balance["taux_positifs"] < 0.5  # classe minoritaire : class_weight=balanced


def test_validate_features_detects_problems():
    bad = pd.DataFrame({c: [1.0] for c in FEATURE_COLUMNS})
    bad.loc[0, "age_client"] = 250
    bad.loc[0, "prime_annuelle"] = np.nan
    problems = validate_features(bad)["problemes"]
    assert any("age_client" in p for p in problems)
    assert any("prime_annuelle" in p for p in problems)


def test_population_drift_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(50, 10, 5000))
    assert population_drift(ref, pd.Series(rng.normal(50, 10, 5000))) < 0.1
    assert population_drift(ref, pd.Series(rng.normal(70, 10, 5000))) > 0.25


def test_train_uses_data_lake_when_available(tmp_path, monkeypatch):
    import src.ml.lake_dataset as ld
    import src.ml.train_model as tm
    generate_lake(n_clients=3000, seed=42, lake_dir=tmp_path)
    monkeypatch.setattr(ld, "LAKE_DIR", tmp_path)
    monkeypatch.setattr(tm, "load_lake_tables", lambda: ld.load_lake_tables(tmp_path))
    monkeypatch.setattr(tm, "MODEL_DIR", tmp_path / "artifacts")
    metrics = tm.train()
    assert metrics["source"] == "data_lake_parquet"
    assert 0.6 < metrics["roc_auc"] < 0.95  # crédible : ni hasard, ni 'trop beau pour etre vrai'
