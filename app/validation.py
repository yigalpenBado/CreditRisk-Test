"""Module de validation : contrat d'entrée d'une demande de crédit.

Deux comportements distincts (plan de test) :
- REJET (validate_data, ValueError avec un message précis) : entrée invalide ou incomplète
  (EX01 à EX03, EX06, EX08) ;
- ALERTE sans rejet (check_input) : entrée valide mais inhabituelle ou incohérente (T11).
"""

import numpy as np
import pandas as pd

ID_COLUMN = "CustomerID"  # identifiant : exclu du modèle et des journaux
TARGET = "Risk"  # "Risk" -> defaut = 1, "No Risk" -> defaut = 0
TARGET_LABELS = {"No Risk": 0, "Risk": 1}

# Contrat d'entrée d'une demande (plan de test, EX01 à EX03)
AGE_RANGE = (18, 75)  # EX01 : hors intervalle -> rejet
DURATION_RANGE = (1, 72)  # EX03 : entier, en mois
# EX02 : revenu mensuel, nombre fini > 0. Colonne ajoutée au jeu de données, calculée par :
#   mensualité = LoanAmount * r(1+r)^n / ((1+r)^n - 1), r = 10 % / 12, n = LoanDuration
#   revenu = mensualité / part du revenu consacrée à la mensualité, arrondi à 2 décimales
INCOME_COLUMN = "MonthlyIncome"
ANNUAL_RATE = 0.10
# Code InstallmentPercent -> part du revenu (centre des tranches du German Credit : 1 = 35 % et plus,
# 2 = 25 à 35 %, 3 = 20 à 25 %, 4 = moins de 20 % ; codes 5 et 6, absents de l'original, prolongés)
INSTALLMENT_SHARE = {1: 0.35, 2: 0.30, 3: 0.225, 4: 0.175, 5: 0.125, 6: 0.075}
INCOME_TOLERANCE = 0.01  # écart toléré (arrondi) avant une alerte d'incohérence du revenu

NUMERIC_FEATURES = [
    "LoanDuration", "LoanAmount", "InstallmentPercent", "CurrentResidenceDuration",
    "Age", "ExistingCreditsCount", "Dependents", INCOME_COLUMN,
]
# Variables numériques entières de valeur >= 1 (en plus de Age, LoanDuration, InstallmentPercent)
COUNT_FEATURES = ["CurrentResidenceDuration", "ExistingCreditsCount", "Dependents"]
# Modalités autorisées (relevées sur le jeu de données) : toute autre valeur est rejetée
CATEGORIES = {
    "CheckingStatus": ["0_to_200", "greater_200", "less_0", "no_checking"],
    "CreditHistory": [
        "all_credits_paid_back", "credits_paid_to_date", "no_credits",
        "outstanding_credit", "prior_payments_delayed",
    ],
    "LoanPurpose": [
        "appliances", "business", "car_new", "car_used", "education", "furniture",
        "other", "radio_tv", "repairs", "retraining", "vacation",
    ],
    "ExistingSavings": ["100_to_500", "500_to_1000", "greater_1000", "less_100", "unknown"],
    "EmploymentDuration": ["1_to_4", "4_to_7", "greater_7", "less_1", "unemployed"],
    "Sex": ["female", "male"],
    "OthersOnLoan": ["co-applicant", "guarantor", "none"],
    "OwnsProperty": ["car_other", "real_estate", "savings_insurance", "unknown"],
    "InstallmentPlans": ["bank", "none", "stores"],
    "Housing": ["free", "own", "rent"],
    "Job": ["management_self-employed", "skilled", "unemployed", "unskilled"],
    "Telephone": ["none", "yes"],
    "ForeignWorker": ["no", "yes"],
}
CATEGORICAL_FEATURES = list(CATEGORIES)
# Les 21 variables explicatives, dans l'ordre du fichier (la cible Risk précède le revenu)
FEATURES = [
    "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose", "LoanAmount",
    "ExistingSavings", "EmploymentDuration", "InstallmentPercent", "Sex", "OthersOnLoan",
    "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans", "Housing",
    "ExistingCreditsCount", "Job", "Dependents", "Telephone", "ForeignWorker", INCOME_COLUMN,
]


def expected_income(amount, duration, installment_percent):
    """Revenu mensuel attendu d'après la formule utilisée pour construire le jeu de données."""
    amount = np.asarray(amount, dtype=float)
    duration = np.asarray(duration, dtype=float)
    share = pd.Series(np.atleast_1d(installment_percent)).map(INSTALLMENT_SHARE).to_numpy(dtype=float)
    r = ANNUAL_RATE / 12
    monthly_payment = amount * r * (1 + r) ** duration / ((1 + r) ** duration - 1)
    return monthly_payment / share


def _is_integer(values):
    return bool(np.all(values == np.round(values)))


def validate_data(df, require_target=True):
    """Contrôle le format et la cohérence ; lève ValueError si le contrat n'est pas respecté.

    La cible est contrôlée si require_target=True ou si elle est présente.
    L'identifiant et les colonnes en trop sont tolérés (ignorés par preprocess_data).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Les données doivent être un DataFrame non vide.")
    if df.columns.duplicated().any():
        raise ValueError(f"Colonne dupliquée : {', '.join(df.columns[df.columns.duplicated()])}")
    with_target = require_target or TARGET in df.columns
    missing = [c for c in FEATURES + ([TARGET] if with_target else []) if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne manquante : {', '.join(missing)}")

    # Valeurs manquantes (EX06), puis types et valeurs finies des variables numériques (T02)
    for col in FEATURES + ([TARGET] if with_target else []):
        if df[col].isna().any():
            raise ValueError(f"Valeur manquante dans {col}")
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            raise ValueError(f"Colonne non numérique : {col}")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Valeur infinie dans {col}")

    # Intervalles et types entiers (EX01 à EX03, EX08)
    age = df["Age"].to_numpy(dtype=float)
    if not _is_integer(age):
        raise ValueError("Age doit être un entier")
    if ((age < AGE_RANGE[0]) | (age > AGE_RANGE[1])).any():
        raise ValueError(f"Age hors de l'intervalle autorisé [{AGE_RANGE[0]}, {AGE_RANGE[1]}]")
    duration = df["LoanDuration"].to_numpy(dtype=float)
    if not _is_integer(duration) or ((duration < DURATION_RANGE[0]) | (duration > DURATION_RANGE[1])).any():
        raise ValueError(f"LoanDuration doit être un entier entre {DURATION_RANGE[0]} et {DURATION_RANGE[1]}")
    for col in ("LoanAmount", INCOME_COLUMN):
        if (df[col] <= 0).any():
            raise ValueError(f"{col} doit être strictement positif")
    if not df["InstallmentPercent"].isin(list(INSTALLMENT_SHARE)).all():
        raise ValueError("InstallmentPercent doit être un code entier entre 1 et 6")
    for col in COUNT_FEATURES:
        values = df[col].to_numpy(dtype=float)
        if not _is_integer(values) or (values < 1).any():
            raise ValueError(f"{col} doit être un entier supérieur ou égal à 1")

    # Modalités des variables catégorielles et libellés de la cible (EX08)
    for col, allowed in CATEGORIES.items():
        unknown = sorted(set(df[col].astype(str)) - set(allowed))
        if unknown:
            raise ValueError(f"Modalité inconnue pour {col} : {', '.join(unknown)}")
    if with_target:
        unknown = sorted(set(df[TARGET].astype(str)) - set(TARGET_LABELS))
        if unknown:
            raise ValueError(f"Libellé de cible inconnu : {', '.join(unknown)} (attendu : Risk ou No Risk)")
    return df


# Alertes sans rejet (T11) : entrées valides mais inhabituelles ou incohérentes
DRIFT_THRESHOLD = 0.5  # écart de médiane toléré, en écarts-types du train
MIN_ROWS_DRIFT = 100  # en dessous, la médiane d'un lot est trop instable pour juger une dérive
EMPLOYED_SINCE_1_YEAR = ["1_to_4", "4_to_7", "greater_7"]


def check_input(X_ref, X):
    """Retourne la liste des alertes sur un lot de demandes X (liste vide : rien à signaler).

    X_ref : données d'entraînement servant de référence (plages et médianes).
    """
    alerts = []
    # Valeurs hors de la plage observée à l'entraînement
    for col in NUMERIC_FEATURES:
        low, high = X_ref[col].min(), X_ref[col].max()
        n_out = int(((X[col] < low) | (X[col] > high)).sum())
        if n_out:
            alerts.append(f"{col} hors de la plage d'entraînement [{low}, {high}] : {n_out} demande(s)")
    # Incohérence entre champs : sans emploi mais employé depuis plus d'un an
    n_job = int(((X["Job"] == "unemployed") & X["EmploymentDuration"].isin(EMPLOYED_SINCE_1_YEAR)).sum())
    if n_job:
        alerts.append(
            f"Incohérence d'emploi (Job = unemployed avec une ancienneté d'emploi) : {n_job} demande(s)")
    # Incohérence du revenu avec le montant, la durée et InstallmentPercent
    gap = np.abs(expected_income(X["LoanAmount"], X["LoanDuration"], X["InstallmentPercent"])
                 - X[INCOME_COLUMN].to_numpy(dtype=float))
    n_income = int((gap > INCOME_TOLERANCE).sum())
    if n_income:
        alerts.append(f"Incohérence du revenu avec la mensualité : {n_income} demande(s)")
    # Dérive : médiane éloignée de celle du train (la médiane résiste à une valeur extrême isolée)
    if len(X) >= MIN_ROWS_DRIFT:
        for col in NUMERIC_FEATURES:
            gap = abs(X[col].median() - X_ref[col].median()) / X_ref[col].std()
            if gap > DRIFT_THRESHOLD:
                alerts.append(f"Dérive possible sur {col} : médiane à {gap:.2f} écart-type du train")
    return alerts
