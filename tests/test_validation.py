"""Tests unitaires de la validation, du prétraitement et de l'évaluation (T01 à T20).

Chaque cas de rejet modifie une seule valeur d'une demande valide et vérifie le message exact.

Ce fichier définit aussi les données de test partagées (jeux synthétiques, graines fixes), importées
par les autres fichiers de tests : les demandes générées respectent le contrat d'entrée (âge 19 à
74 ans, durée 4 à 60 mois, modalités autorisées, revenu calculé par la formule du jeu de données,
donc aucune alerte).
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, reload_model
from app.prediction import choose_threshold, evaluate_model, save_model, train_model
from app.preprocessing import DATA_PATH, load_data, preprocess_data, split_data
from app.validation import (
    CATEGORIES,
    FEATURES,
    ID_COLUMN,
    INCOME_COLUMN,
    NUMERIC_FEATURES,
    TARGET,
    TARGET_LABELS,
    check_input,
    expected_income,
    validate_data,
)

# --- Données de test partagées ---

def make_requests(n, seed):
    """n demandes de crédit valides et cohérentes, dans l'ordre des colonnes du contrat."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(values, n) for col, values in CATEGORIES.items()})
    # Pas de « Job = unemployed » : il déclencherait une alerte d'incohérence avec l'ancienneté
    df["Job"] = rng.choice(["management_self-employed", "skilled", "unskilled"], n)
    df["LoanDuration"] = rng.integers(4, 61, n)
    df["LoanAmount"] = rng.integers(250, 11_000, n)
    df["InstallmentPercent"] = rng.integers(1, 7, n)
    df["CurrentResidenceDuration"] = rng.integers(1, 5, n)
    df["Age"] = rng.integers(19, 75, n)
    df["ExistingCreditsCount"] = rng.integers(1, 4, n)
    df["Dependents"] = rng.integers(1, 3, n)
    income = expected_income(df["LoanAmount"], df["LoanDuration"], df["InstallmentPercent"])
    df[INCOME_COLUMN] = income.round(2)
    return df[FEATURES]


@pytest.fixture
def valid_df():
    """Petit jeu conforme au contrat : 20 demandes dont 6 défauts, avec identifiant et cible."""
    df = make_requests(20, seed=42)
    df.insert(0, ID_COLUMN, [f"id-{i}" for i in range(20)])
    df[TARGET] = ["No Risk"] * 14 + ["Risk"] * 6
    return df


@pytest.fixture(scope="module")
def train_data():
    """Jeu déséquilibré (30 % de défauts) où les défauts ont des durées plus longues (graine 0).

    Retourne (X, y) avec y en 0/1 ; scope="module" : généré une fois par fichier, non modifié.
    """
    n, n_default = 600, 180
    X = make_requests(n, seed=0)
    y = pd.Series([0] * (n - n_default) + [1] * n_default)
    rng = np.random.default_rng(1)
    X.loc[y == 1, "LoanDuration"] = rng.integers(36, 73, n_default)  # signal séparateur
    X.loc[y == 0, "LoanDuration"] = rng.integers(4, 37, n - n_default)
    X[INCOME_COLUMN] = expected_income(X["LoanAmount"], X["LoanDuration"], X["InstallmentPercent"]).round(2)
    return X, y


@pytest.fixture(scope="module")
def model(train_data):
    """Régression logistique entraînée sur train_data."""
    return train_model(*train_data)


# Demande valide et cohérente envoyée à l'API (revenu calculé par la formule du jeu de données)
DEMANDE = {
    "CheckingStatus": "0_to_200", "LoanDuration": 12, "CreditHistory": "credits_paid_to_date",
    "LoanPurpose": "car_new", "LoanAmount": 1000.0, "ExistingSavings": "100_to_500",
    "EmploymentDuration": "1_to_4", "InstallmentPercent": 3, "Sex": "female", "OthersOnLoan": "none",
    "CurrentResidenceDuration": 2, "OwnsProperty": "real_estate", "Age": 35, "InstallmentPlans": "none",
    "Housing": "own", "ExistingCreditsCount": 1, "Job": "skilled", "Dependents": 1,
    "Telephone": "none", "ForeignWorker": "yes",
    "MonthlyIncome": round(float(expected_income(1000, 12, 3)[0]), 2),
}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory, model, train_data):
    """Modèle synthétique sauvegardé, seuil 0,4."""
    path = tmp_path_factory.mktemp("modele") / "model.pkl"
    save_model(model, 0.4, path, reference=train_data[0])
    return path


@pytest.fixture
def client(model_path, monkeypatch):
    """Client de l'API avec le modèle synthétique chargé."""
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    reload_model()
    return TestClient(app)


# --- Tests de validation et de prétraitement ---

def with_value(df, col, value):
    """Copie de df dont la première ligne vaut `value` dans la colonne `col`."""
    df = df.copy()
    df[col] = df[col].astype(object)
    df.loc[0, col] = value
    return df.infer_objects()


def test_load_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Fichier de données introuvable"):
        load_data(tmp_path / "absent.csv")
    if DATA_PATH.exists():  # le jeu réel respecte le contrat d'entrée
        assert len(validate_data(load_data())) == 5000


# T02, T04, T06 : demande valide et valeurs limites acceptées (EX01, EX02, EX03)
def test_valeurs_limites_acceptees(valid_df):
    assert validate_data(valid_df) is valid_df
    for col, value in [("Age", 18), ("Age", 75), ("LoanDuration", 1), ("LoanDuration", 72),
                       (INCOME_COLUMN, 0.01)]:
        validate_data(with_value(valid_df, col, value))


# T01, T03, T05, T07, T08, T10 : une seule valeur fautive, message explicite
@pytest.mark.parametrize("col, value, message", [
    ("Age", 17, r"Age hors de l'intervalle autorisé \[18, 75\]"),         # EX01
    ("Age", 76, r"Age hors de l'intervalle autorisé \[18, 75\]"),
    (INCOME_COLUMN, "abc", "Colonne non numérique : MonthlyIncome"),      # EX02
    (INCOME_COLUMN, 0.0, "MonthlyIncome doit être strictement positif"),
    ("LoanDuration", 12.5, "LoanDuration doit être un entier entre 1 et 72"),  # EX03
    ("LoanAmount", np.nan, "Valeur manquante dans LoanAmount"),           # EX06
    ("Sex", "unknown", "Modalité inconnue pour Sex : unknown"),           # EX08
    (TARGET, "Maybe", "Libellé de cible inconnu : Maybe"),
])
def test_valeur_invalide_rejetee(valid_df, col, value, message):
    with pytest.raises(ValueError, match=message):
        validate_data(with_value(valid_df, col, value))


def test_colonne_manquante_rejetee(valid_df):  # T07 (EX06)
    with pytest.raises(ValueError, match="Colonne manquante : Age"):
        validate_data(valid_df.drop(columns="Age"))


# T11 : alertes sans rejet sur une demande valide mais inhabituelle ou incohérente (EX08)
def test_alertes(valid_df):
    assert check_input(valid_df, valid_df) == []
    demande = valid_df.head(1).assign(LoanAmount=10**6, Job="unemployed", EmploymentDuration="greater_7")
    alertes = " | ".join(check_input(valid_df, demande))
    assert "LoanAmount hors de la plage" in alertes
    assert "Incohérence d'emploi" in alertes
    assert "Incohérence du revenu" in alertes


# T12, T13 : doublons supprimés, cible codée, partitions disjointes et stratifiées (EX09)
def test_preparation_et_decoupage(valid_df, train_data):
    X, y = preprocess_data(pd.concat([valid_df, valid_df.head(3)]))
    assert list(X.columns) == FEATURES and len(X) == 20  # 3 doublons supprimés
    assert y.tolist() == [0] * 14 + [1] * 6  # Risk -> 1, No Risk -> 0

    X, y = train_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    assert (len(X_train), len(X_val), len(X_test)) == (360, 120, 120)
    assert not set(X_train.index) & set(X_val.index) and not set(X_train.index) & set(X_test.index)
    assert all(abs(part.mean() - y.mean()) <= 0.01 for part in (y_train, y_val, y_test))


# T15 : seuil de meilleure précision parmi ceux qui atteignent le rappel visé, sur la validation
def test_choix_du_seuil():
    y_val, proba_val = np.array([0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    assert choose_threshold(y_val, proba_val, min_recall=1.0) == 0.70


# T20 : métriques adaptées au déséquilibre, pas l'accuracy seule (EX10)
def test_metriques():
    m = evaluate_model(np.array([0, 0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.4]), 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 3)
    assert (m["recall"], m["precision"], m["fpr"]) == (0.5, 0.5, 0.25)
    assert m["roc_auc"] == pytest.approx(0.875)


# --- Exploration du jeu réel (pytest -m mesures) : figures reports/eda_*.png ---

@pytest.mark.mesures
@pytest.mark.skipif(not DATA_PATH.exists(), reason="jeu de données réel absent")
def test_exploration_du_jeu_reel():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = DATA_PATH.parents[2] / "reports"
    df = load_data()
    y = df[TARGET].map(TARGET_LABELS)

    # 1. Schéma, qualité : les 5 000 lignes respectent le contrat ; aucune valeur manquante ni infinie,
    #    identifiants uniques ; 18 demandes identiques sous un autre identifiant (supprimées avant le
    #    découpage par preprocess_data, sinon fuite entre train et test)
    assert len(validate_data(df)) == 5000
    assert df.isna().sum().sum() == 0 and not np.isinf(df[NUMERIC_FEATURES]).any().any()
    assert not df[ID_COLUMN].duplicated().any()
    assert df.drop(columns=ID_COLUMN).duplicated().sum() == 18

    # 2. Déséquilibre : 33,4 % de défauts ; prédire toujours « pas de défaut » donne 66,6 % d'accuracy
    #    sans détecter aucun défaut, d'où l'évaluation par rappel, précision, PR-AUC (T20)
    assert y.mean() == pytest.approx(0.334, abs=0.001)
    fig, ax = plt.subplots(figsize=(4, 3))
    y.map({0: "No Risk", 1: "Risk"}).value_counts().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Répartition de la cible")
    ax.set_ylabel("Demandes")
    fig.tight_layout()
    fig.savefig(reports / "eda_cible.png", dpi=120)

    # 3. Variables numériques : effet positif sur le défaut pour toutes (d de Cohen), fort (> 0,8) pour
    #    l'âge, le montant et la durée ; le revenu (calculé) est très asymétrique, d'où log(1 + x)
    pooled = np.sqrt((df.loc[y == 1, NUMERIC_FEATURES].var() + df.loc[y == 0, NUMERIC_FEATURES].var()) / 2)
    d_cohen = (df.loc[y == 1, NUMERIC_FEATURES].mean() - df.loc[y == 0, NUMERIC_FEATURES].mean()) / pooled
    assert (d_cohen > 0).all() and (d_cohen[["Age", "LoanAmount", "LoanDuration"]] > 0.8).all()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, col in zip(axes, ["Age", "LoanAmount", "LoanDuration", INCOME_COLUMN]):
        for label, color in ((0, "#4C72B0"), (1, "#C44E52")):
            ax.hist(df.loc[y == label, col], bins=30, density=True, alpha=0.5, color=color,
                    label="Risk" if label else "No Risk")
        ax.set_title(col)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(reports / "eda_distributions.png", dpi=120)

    # 4. Attribut sensible : le taux de défaut croît avec l'âge (7 % de 19 à 25 ans, 78 % après 50 ans),
    #    biais volontaire des données, que le modèle risque de reproduire
    par_age = y.groupby(pd.cut(df["Age"], [18, 25, 35, 50, 75]), observed=True).mean()
    assert par_age.iloc[0] < 0.10 and par_age.iloc[-1] > 0.75
    fig, ax = plt.subplots(figsize=(5, 3))
    par_age.plot.bar(ax=ax, color="#C44E52")
    ax.set_title("Taux de défaut par tranche d'âge")
    ax.set_ylabel("Taux de défaut")
    fig.tight_layout()
    fig.savefig(reports / "eda_age.png", dpi=120)

    # 5. Corrélations : variables explicatives fortement corrélées entre elles (revenu calculé à partir
    #    du montant, de la durée et d'InstallmentPercent) : coefficients non interprétables un à un
    corr = df[NUMERIC_FEATURES].assign(defaut=y).corr()
    assert corr.loc["LoanAmount", INCOME_COLUMN] > 0.7
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im)
    ax.set_title("Corrélations (variables numériques et défaut)")
    fig.tight_layout()
    fig.savefig(reports / "eda_correlations.png", dpi=120)
    plt.close("all")

    # 6. Seule alerte sur le jeu complet : Job = unemployed avec plus d'un an d'ancienneté (alerte sans
    #    rejet, T11) ; aucune incohérence du revenu
    alertes = check_input(df, df)
    assert len(alertes) == 1 and "unemployed" in alertes[0]
