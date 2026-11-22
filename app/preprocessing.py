"""Module de prétraitement : chargement, préparation et découpage des données.

Corrections documentées appliquées par preprocess_data : doublons supprimés, identifiant et
colonnes en trop ignorés, colonnes remises dans l'ordre attendu, cible convertie en 0/1.
La standardisation, le log du revenu et l'encodage one-hot sont faits dans le Pipeline
(app/prediction.py), donc ajustés sur le train uniquement (T14).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.validation import FEATURES, TARGET, TARGET_LABELS, validate_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "german_credit_data.csv"
SEED = 42


def load_data(path=DATA_PATH):
    """Lit le fichier CSV ; FileNotFoundError s'il est absent, ValueError s'il est vide."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Le fichier de données est vide : {path}")
    return df


def preprocess_data(df, require_target=True):
    """Valide puis prépare les données ; retourne (X, y), avec y = None sans cible.

    Les doublons sont supprimés (hors identifiant) : une demande en double fausserait l'évaluation.
    """
    validate_data(df, require_target)
    if TARGET not in df.columns:
        return df[FEATURES].reset_index(drop=True), None
    data = df[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
    return data[FEATURES], data[TARGET].map(TARGET_LABELS).astype(int)


def split_data(X, y, seed=SEED):
    """Découpage stratifié 60/20/20 : (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
