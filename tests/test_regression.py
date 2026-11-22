"""Tests de non-régression (Partie V) : comparer la version courante à la référence figée.

Référence : reports/baseline.json, créée par `python -m app.prediction --save-baseline` et jamais
écrasée. Tolérances fixées avant la comparaison (app/prediction.py) :
- baisse du rappel ou du F1 > 0,02 : version rejetée ;
- baisse de la précision ou de la ROC-AUC > 0,02, probabilité modifiée de plus de 0,05,
  données ou bibliothèques différentes : investigation ;
- écart avant / après sauvegarde et rechargement : 1e-12 au plus.
Les tests fonctionnels (test_api, test_integration…) sont rejoués sur chaque version par la CI.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.prediction import (
    MAIN_MODEL,
    MODEL_PATH,
    MODELS,
    PROBA_TOLERANCE,
    RELOAD_TOLERANCE,
    VAL_MIN_RECALL,
    compare_to_baseline,
    describe_version,
    evaluate_model,
    load_model,
    predict,
    save_model,
    train_and_validate,
)
from app.preprocessing import DATA_PATH, SEED, load_data, preprocess_data, split_data
from app.validation import CATEGORIES, INCOME_COLUMN, NUMERIC_FEATURES

BASELINE = Path(__file__).resolve().parents[1] / "reports" / "baseline.json"
# Ces tests utilisent le jeu réel data/raw/german_credit_data.csv : s'il n'est pas versionné (absent
# du dépôt, donc de la CI), ils sont ignorés, et les autres tests utilisent des données synthétiques
reel = pytest.mark.skipif(not (BASELINE.exists() and MODEL_PATH.exists() and DATA_PATH.exists()),
                          reason="jeu de données réel, modèle ou référence absent")


def version(recall=0.80, f1=0.70, precision=0.60, roc_auc=0.87, proba=(0.1, 0.5), data="a", libs="1"):
    return {"validation_metrics": {"recall": recall, "f1": f1, "precision": precision, "roc_auc": roc_auc},
            "validation_proba": list(proba), "data_sha256": data, "libraries": {"scikit-learn": libs}}


# Règle de décision : une variation dans les tolérances est acceptée ; rejet si le rappel ou le F1
# baisse trop ; investigation pour les autres écarts (ici, une probabilité modifiée de plus de 0,05)
@pytest.mark.parametrize("courante, decision", [
    (version(recall=0.79, f1=0.69, proba=(0.14, 0.46)), "accepter"),  # dans les tolérances
    (version(recall=0.77), "rejeter"),                                  # rappel - 0,03
    (version(f1=0.67), "rejeter"),                                      # F1 - 0,03
    (version(proba=(0.1, 0.56)), "investiguer"),                        # probabilité + 0,06
])
def test_regle_de_decision(courante, decision):
    assert compare_to_baseline(version(), courante)[0] == decision


@pytest.fixture(scope="module")
def donnees_reelles():
    X, y = preprocess_data(load_data())
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)
    return X_train, y_train, X_val, y_val


# Métriques de la version courante du code comparées à la référence : le test échoue si la
# version doit être rejetée (baisse du rappel ou du F1 au-delà de la tolérance)
@reel
def test_metriques_comparees_a_la_reference(donnees_reelles):
    X_train, y_train, X_val, y_val = donnees_reelles
    model, threshold, proba_val = train_and_validate(X_train, y_train, X_val, y_val)
    courante = describe_version(model, "logistic_regression", threshold,
                                evaluate_model(y_val, proba_val, threshold), proba_val, DATA_PATH)
    decision, motifs = compare_to_baseline(json.loads(BASELINE.read_text()), courante)
    assert decision != "rejeter", motifs


# Modèle sauvegardé : sorties conformes au contrat, identiques après sauvegarde et rechargement
# (1e-12), même seuil et probabilités à moins de 0,05 de la référence
@reel
def test_modele_sauvegarde_coherent(donnees_reelles, tmp_path):
    X_val = donnees_reelles[2]
    model, threshold = load_model(MODEL_PATH)
    proba, decisions = predict(model, X_val, threshold)
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in proba)
    assert decisions.tolist() == (proba >= threshold).astype(int).tolist()

    save_model(model, threshold, tmp_path / "copie.pkl")
    copie, seuil_copie = load_model(tmp_path / "copie.pkl")
    assert seuil_copie == threshold
    assert np.abs(copie.predict_proba(X_val)[:, 1] - proba).max() <= RELOAD_TOLERANCE

    reference = json.loads(BASELINE.read_text())
    assert threshold == reference["threshold"]
    assert np.abs(proba - reference["validation_proba"]).max() <= PROBA_TOLERANCE


# --- Mesures sur le jeu réel (pytest -m mesures), résultats écrits dans reports/ ---

def pipeline_candidat(model, categories=None, log_income=True):
    """Pipeline de la référence, avec ou sans log du revenu, sur un sous-ensemble de catégories."""
    categories = categories or CATEGORIES
    log = MODELS[MAIN_MODEL].named_steps["prep"].transformers[0][1]
    transformers = [("log", log, [INCOME_COLUMN])] if log_income else []
    num = [c for c in NUMERIC_FEATURES if c != INCOME_COLUMN or not log_income]
    transformers += [("num", StandardScaler(), num),
                     ("cat", OneHotEncoder(categories=list(categories.values())), list(categories))]
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", model)])


LR = dict(max_iter=1000, random_state=SEED)
SIMPLIFIE = {k: v for k, v in CATEGORIES.items() if k not in ("CheckingStatus", "CreditHistory")}
# Versions candidates : chacune représente une modification du code du modèle -> (pipeline, rappel
# visé sur la validation, décision attendue de la règle de non-régression)
CANDIDATES = {
    "V0 : référence rejouée (aucune modification)": (MODELS[MAIN_MODEL], VAL_MIN_RECALL, "accepter"),
    "V1 : revenu sans transformation log": (
        pipeline_candidat(LogisticRegression(**LR), log_income=False), VAL_MIN_RECALL, "accepter"),
    "V2 : pondération des classes (balanced)": (
        pipeline_candidat(LogisticRegression(class_weight="balanced", **LR)), VAL_MIN_RECALL, "investiguer"),
    "V3 : sans CheckingStatus ni CreditHistory": (
        pipeline_candidat(LogisticRegression(**LR), SIMPLIFIE), VAL_MIN_RECALL, "investiguer"),
    "V4 : régularisation excessive (C = 0,001)": (
        pipeline_candidat(LogisticRegression(C=0.001, **LR)), VAL_MIN_RECALL, "rejeter"),
    "V5 : erreur de code, rappel visé 0,60": (MODELS[MAIN_MODEL], 0.60, "rejeter"),
}


# Chaque candidate est entraînée sur les mêmes données et le même découpage que la référence, son
# seuil choisi sur la validation, puis comparée à reports/baseline.json. Ni le modèle sauvegardé ni
# la référence ne sont modifiés. Résultats : reports/non_regression_candidats.csv et .json
@pytest.mark.mesures
@reel
def test_versions_candidates(donnees_reelles):
    X_train, y_train, X_val, y_val = donnees_reelles
    baseline = json.loads(BASELINE.read_text())
    rows, details = [], {}
    for name, (candidate, min_recall, attendue) in CANDIDATES.items():
        model, threshold, proba = train_and_validate(X_train, y_train, X_val, y_val, pipeline=candidate,
                                                     min_recall=min_recall)
        metrics = evaluate_model(y_val, proba, threshold)
        decision, reasons = compare_to_baseline(
            baseline, describe_version(model, name, threshold, metrics, proba, DATA_PATH))
        gap = np.abs(np.array(proba) - baseline["validation_proba"])
        rows.append({"version": name, "seuil": threshold,
                     **{k: round(metrics[k] - baseline["validation_metrics"][k], 4)
                        for k in ("recall", "f1", "precision", "roc_auc", "brier")},
                     "proba_ecart_max": round(float(gap.max()), 4),
                     "proba_ecart_median": round(float(np.median(gap)), 4), "decision": decision})
        details[name] = {"seuil": threshold, "metriques": {k: round(v, 4) for k, v in metrics.items()},
                         "decision": decision, "motifs": reasons}
    reports = BASELINE.parent
    pd.DataFrame(rows).to_csv(reports / "non_regression_candidats.csv", index=False)
    (reports / "non_regression_candidats.json").write_text(json.dumps(
        {"reference": {"code": baseline["code"], "seuil": baseline["threshold"],
                       "metriques": {k: round(v, 4) for k, v in baseline["validation_metrics"].items()}},
         "candidates": details}, indent=2, ensure_ascii=False))
    assert {n: r["decision"] for n, r in details.items()} == {n: c[2] for n, c in CANDIDATES.items()}
