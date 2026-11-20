"""
Construction du dataset d'entraînement à partir du data lake (Parquet), sans fuite de données.

Principe anti-leakage (point de coupure temporel) :
- `observation_date` : date à laquelle on "prédit". Les variables (features) ne regardent QUE
  ce qui est connu avant cette date (contrats commencés, sinistres et paiements antérieurs).
- La cible `risque_client` est calculée sur la fenêtre FUTURE (observation_date ->
  observation_date + horizon) : le client a-t-il déclaré un sinistre important ou subi un
  impayé dans l'horizon ?
- Des colonnes comme `date_fin` du contrat ou `statut_contrat` final ne sont JAMAIS des
  features : elles ne sont pas connues au moment de prédire (cf. LEAKY_COLUMNS).

Le data lake contient clients.parquet (produit par le job Spark) ainsi que contrats, sinistres
et paiements consolidés (même convention source_system / source_id que le MCD).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAKE_DIR = Path(__file__).resolve().parents[2] / "data" / "hdfs_sim"

# Colonnes interdites comme variables explicatives : elles révèlent l'issue (data leakage).
LEAKY_COLUMNS = frozenset(
    {"date_fin", "statut_contrat", "statut_sinistre", "statut_transaction", "score_fraude",
     "risque_client"}
)

FEATURE_COLUMNS = [
    "anciennete_contrat_mois",
    "nb_sinistres_12m",
    "montant_total_sinistres",
    "nb_paiements_retard",
    "age_client",
    "prime_annuelle",
]

HIGH_CLAIM_THRESHOLD = 8000.0  # montant (EUR) au-delà duquel un sinistre est "important"
MIN_FUTURE_LATE_PAYMENTS = 3  # nb de retards dans l'horizon à partir duquel on parle d'impayés


class DataLeakageError(ValueError):
    """Levée quand une variable interdite (non connue au moment de prédire) est utilisée."""


def assert_no_leakage(feature_columns) -> None:
    """Garde-fou : refuse toute variable qui révèle la cible ou le futur."""
    leaked = sorted(set(feature_columns) & LEAKY_COLUMNS)
    if leaked:
        raise DataLeakageError(f"Variables interdites (fuite de données) : {leaked}")


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def build_training_frame(
    clients: pd.DataFrame,
    contrats: pd.DataFrame,
    sinistres: pd.DataFrame,
    paiements: pd.DataFrame,
    observation_date: str | pd.Timestamp,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """Une ligne par client ; features = passé, cible = futur (aucune fuite temporelle)."""
    obs = pd.Timestamp(observation_date)
    end = obs + pd.DateOffset(months=horizon_months)
    lookback = obs - pd.DateOffset(months=12)

    contrats = contrats.assign(date_debut=_to_datetime(contrats["date_debut"]))
    sinistres = sinistres.assign(date_sinistre=_to_datetime(sinistres["date_sinistre"]))
    paiements = paiements.assign(date_paiement=_to_datetime(paiements["date_paiement"]))

    # --- Features : uniquement des événements STRICTEMENT antérieurs à observation_date ---
    c_past = contrats[contrats["date_debut"] < obs]
    per_client = c_past.groupby("client_id").agg(
        debut_min=("date_debut", "min"), prime_annuelle=("prime_annuelle", "sum")
    )

    c_map = contrats[["contrat_id", "client_id"]]
    s = sinistres.merge(c_map, on="contrat_id", how="inner")
    p = paiements.merge(c_map, on="contrat_id", how="inner")

    s_past = s[(s["date_sinistre"] < obs) & (s["date_sinistre"] >= lookback)]
    s_feat = s_past.groupby("client_id").agg(
        nb_sinistres_12m=("sinistre_id", "count"), montant_total_sinistres=("montant_estime", "sum")
    )
    p_late_past = p[(p["date_paiement"] < obs) & (p["statut_transaction"] == "RETARD")]
    p_feat = p_late_past.groupby("client_id").agg(nb_paiements_retard=("paiement_id", "count"))

    df = clients[["client_id", "date_naissance"]].merge(per_client, on="client_id", how="inner")
    df = df.merge(s_feat, on="client_id", how="left").merge(p_feat, on="client_id", how="left")
    for col in ("nb_sinistres_12m", "montant_total_sinistres", "nb_paiements_retard"):
        df[col] = df[col].fillna(0)

    naissance = _to_datetime(df["date_naissance"])
    df["age_client"] = ((obs - naissance).dt.days // 365).astype("float")
    df["anciennete_contrat_mois"] = ((obs - df["debut_min"]).dt.days // 30).astype("float")

    # --- Cible : événements dans la fenêtre FUTURE [obs, end) ---
    s_fut = s[(s["date_sinistre"] >= obs) & (s["date_sinistre"] < end)]
    gros = s_fut[s_fut["montant_estime"] >= HIGH_CLAIM_THRESHOLD]["client_id"].unique()
    p_fut = p[(p["date_paiement"] >= obs) & (p["date_paiement"] < end)]
    retards_fut = p_fut[p_fut["statut_transaction"] == "RETARD"].groupby("client_id").size()
    multi_impayes = retards_fut[retards_fut >= MIN_FUTURE_LATE_PAYMENTS].index
    df["risque_client"] = df["client_id"].isin(set(gros) | set(multi_impayes)).astype(int)

    out = df[FEATURE_COLUMNS + ["risque_client"]].dropna()
    assert_no_leakage(FEATURE_COLUMNS)
    return out.reset_index(drop=True)


def load_lake_tables(lake_dir: Path = LAKE_DIR) -> dict[str, pd.DataFrame]:
    names = ("clients", "contrats", "sinistres", "paiements")
    missing = [n for n in names if not (lake_dir / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Tables absentes du data lake : {missing}")
    return {n: pd.read_parquet(lake_dir / f"{n}.parquet") for n in names}


def class_balance(y: pd.Series) -> dict[str, float]:
    """Profil du déséquilibre des classes (à tracer avant d'entraîner)."""
    counts = y.value_counts()
    pos = float(counts.get(1, 0))
    total = float(len(y))
    return {"n": total, "positifs": pos, "taux_positifs": pos / total if total else 0.0}


def validate_features(df: pd.DataFrame) -> dict[str, list[str]]:
    """Validation des features : colonnes, nulls, infinis, plages métier."""
    problems: list[str] = []
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            problems.append(f"colonne absente : {col}")
            continue
        if df[col].isna().any():
            problems.append(f"valeurs nulles : {col}")
        if np.isinf(df[col].astype(float)).any():
            problems.append(f"valeurs infinies : {col}")
    ranges = {"age_client": (16, 110), "anciennete_contrat_mois": (0, 1200),
              "nb_sinistres_12m": (0, 100), "nb_paiements_retard": (0, 1000),
              "montant_total_sinistres": (0, 1e8), "prime_annuelle": (0, 1e6)}
    for col, (lo, hi) in ranges.items():
        if col in df.columns and ((df[col] < lo) | (df[col] > hi)).any():
            problems.append(f"hors plage [{lo}, {hi}] : {col}")
    return {"problemes": problems}
