"""
Entraîne le modèle de **prédiction du risque client** à partir du dataset consolidé
(historique contrats/sinistres/paiements du data lake).

Source des données (par ordre de priorité) :
1. le data lake Parquet (`data/hdfs_sim/`) via `src/ml/lake_dataset.py`, avec séparation
   temporelle features (passé) / cible (futur) pour éviter toute fuite de données ;
2. à défaut (lake absent ou trop petit), un dataset synthétique de repli, explicitement
   signalé dans les métriques (`source = "synthetique"`).

Chaque entraînement est tracé avec MLflow (paramètres, métriques, profil des classes).

Le modèle est un RandomForestClassifier : bon compromis performance / interprétabilité
(feature_importances_) et robuste sur données tabulaires hétérogènes issues de deux
systèmes sources différents.
"""
from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.ml.lake_dataset import (
    FEATURE_COLUMNS as LAKE_FEATURES,
    assert_no_leakage,
    build_training_frame,
    class_balance,
    load_lake_tables,
    validate_features,
)

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURE_COLUMNS = list(LAKE_FEATURES)
MIN_LAKE_ROWS = 200  # en dessous, le lake est jugé trop petit pour entraîner un modèle fiable
OBSERVATION_DATE = "2024-01-01"


def _generate_synthetic_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    anciennete = rng.integers(1, 240, n)
    nb_sinistres = rng.poisson(1.2, n)
    montant_sinistres = np.round(nb_sinistres * rng.gamma(2.0, 800, n), 2)
    nb_retards = rng.poisson(0.5, n)
    age = rng.integers(18, 85, n)
    prime = np.round(rng.normal(650, 200, n).clip(100, 3000), 2)

    # Risque = combinaison logique de sinistralité, retards de paiement et jeunesse du client,
    # + bruit aléatoire (aucun modèle n'est parfaitement déterministe).
    risk_score = (
        0.35 * (nb_sinistres / (nb_sinistres.max() + 1))
        + 0.25 * (montant_sinistres / (montant_sinistres.max() + 1))
        + 0.25 * (nb_retards / (nb_retards.max() + 1))
        + 0.15 * ((85 - age) / 67)
        + rng.normal(0, 0.08, n)
    )
    risque_client = (risk_score > np.quantile(risk_score, 0.7)).astype(int)

    return pd.DataFrame(
        {
            "anciennete_contrat_mois": anciennete,
            "nb_sinistres_12m": nb_sinistres,
            "montant_total_sinistres": montant_sinistres,
            "nb_paiements_retard": nb_retards,
            "age_client": age,
            "prime_annuelle": prime,
            "risque_client": risque_client,
        }
    )


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Charge le dataset depuis le data lake ; repli synthétique si indisponible."""
    try:
        tables = load_lake_tables()
        df = build_training_frame(observation_date=OBSERVATION_DATE, **tables)
        if len(df) >= MIN_LAKE_ROWS and df["risque_client"].nunique() == 2:
            return df, "data_lake_parquet"
    except (FileNotFoundError, KeyError):
        pass
    return _generate_synthetic_dataset(), "synthetique"


def population_drift(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI (Population Stability Index) : mesure la dérive d'une variable entre l'entraînement
    et la production. Règle usuelle : < 0,1 stable ; 0,1-0,25 à surveiller ; > 0,25 dérive."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(np.clip(current, edges[0], edges[-1]), bins=edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _log_mlflow(params: dict, metrics: dict) -> None:
    """Trace l'entraînement dans MLflow (stockage fichier local, aucun serveur requis)."""
    try:
        import mlflow  # type: ignore

        mlflow.set_tracking_uri((MODEL_DIR.parent / "mlruns").as_uri())
        mlflow.set_experiment("risque-client")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception:  # MLflow est optionnel : ne jamais bloquer l'entraînement
        pass


def train() -> dict:
    df, source = load_dataset()
    assert_no_leakage(FEATURE_COLUMNS)
    problems = validate_features(df)["problemes"]
    if problems:
        raise ValueError(f"Features invalides : {problems}")

    X = df[FEATURE_COLUMNS]
    y = df["risque_client"]
    balance = class_balance(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "risk_model.joblib")

    _log_mlflow(
        {"source": source, "n_estimators": 200, "max_depth": 8, "class_weight": "balanced",
         "observation_date": OBSERVATION_DATE},
        {"roc_auc": auc, "taux_positifs": balance["taux_positifs"],
         "n_train": float(len(X_train)), "n_test": float(len(X_test))},
    )

    return {"roc_auc": auc, "report": report, "n_train": len(X_train), "n_test": len(X_test),
            "source": source, "class_balance": balance}


if __name__ == "__main__":
    metrics = train()
    print(f"Modèle entraîné sur {metrics['source']}. ROC AUC = {metrics['roc_auc']:.3f} "
          f"(train={metrics['n_train']}, test={metrics['n_test']})")
