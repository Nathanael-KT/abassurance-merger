"""
Génère un data lake de démonstration complet (clients, contrats, sinistres, paiements) au
format Parquet, conforme au MCD (docs/MCD.md), pour alimenter l'entraînement du modèle.

En production, ces 4 tables sont produites par le job Spark à partir des topics Kafka
(Oracle + SQL Server). Ici, un historique 2022-2024 est simulé : le risque futur dépend du
comportement passé (sinistralité, retards) + du bruit, mais les colonnes "futures" ne sont
jamais utilisées comme variables (cf. lake_dataset.py).
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

LAKE_DIR = Path(__file__).resolve().parents[2] / "data" / "hdfs_sim"
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _uid(system: str, source_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{system}:{source_id}"))


def generate_lake(n_clients: int = 3000, seed: int = 42, lake_dir: Path = LAKE_DIR) -> dict:
    rng = np.random.default_rng(seed)
    lake_dir.mkdir(parents=True, exist_ok=True)

    systems = rng.choice(["ABASSURANCE", "ASSUREPLUS"], n_clients, p=[0.7, 0.3])
    birth = pd.to_datetime("1945-01-01") + pd.to_timedelta(rng.integers(0, 20000, n_clients), "D")
    clients = pd.DataFrame({
        "client_id": [_uid(s, str(i)) for i, s in enumerate(systems)],
        "source_system": systems,
        "source_id": [str(i) for i in range(n_clients)],
        "date_naissance": birth.strftime("%Y-%m-%d"),
        "statut_client": "ACTIF",
    })

    start = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 1800, n_clients), "D")
    risk_latent = rng.beta(2, 5, n_clients)  # propension au risque propre à chaque client
    contrats = pd.DataFrame({
        "contrat_id": [f"C{i}" for i in range(n_clients)],
        "client_id": clients["client_id"],
        "date_debut": start.strftime("%Y-%m-%d"),
        "date_fin": (start + pd.to_timedelta(rng.integers(365, 2500, n_clients), "D")).strftime("%Y-%m-%d"),
        "prime_annuelle": np.round(rng.normal(650, 200, n_clients).clip(100, 3000), 2),
        "statut_contrat": rng.choice(["ACTIF", "RESILIE"], n_clients, p=[0.85, 0.15]),
    })

    # Sinistres et paiements sur 2022-2024 : fréquence pilotée par le risque latent du client.
    rows_s, rows_p = [], []
    sid = pid = 0
    for i in range(n_clients):
        for _ in range(rng.poisson(0.3 + 14 * risk_latent[i])):
            d = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 1100)))
            rows_s.append((sid, f"C{i}", d.strftime("%Y-%m-%d"),
                           round(float(rng.gamma(2.0, 700 + 2500 * risk_latent[i])), 2)))
            sid += 1
        for _ in range(rng.poisson(12)):
            d = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 1100)))
            late = rng.random() < (0.02 + 0.45 * risk_latent[i])
            rows_p.append((pid, f"C{i}", d.strftime("%Y-%m-%d"),
                           round(float(contrats.loc[i, "prime_annuelle"] / 12), 2),
                           "RETARD" if late else "OK"))
            pid += 1
    sinistres = pd.DataFrame(rows_s, columns=["sinistre_id", "contrat_id", "date_sinistre",
                                              "montant_estime"])
    sinistres["statut_sinistre"] = "OUVERT"
    paiements = pd.DataFrame(rows_p, columns=["paiement_id", "contrat_id", "date_paiement",
                                              "montant", "statut_transaction"])

    tables = {"clients": clients, "contrats": contrats, "sinistres": sinistres,
              "paiements": paiements}
    for name, frame in tables.items():
        frame.to_parquet(lake_dir / f"{name}.parquet", index=False)
    return {name: len(frame) for name, frame in tables.items()}


if __name__ == "__main__":
    print("Data lake généré :", generate_lake())
