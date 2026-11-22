"""Modèle de risque de défaut : entraînement, évaluation, sauvegarde, chargement et prédiction.

- entraînement : pipeline (log du revenu, standardisation, one-hot) et régression logistique,
  seuil choisi sur la validation (T10, T15) ;
- évaluation : métriques (T20), description d'une version et règle de non-régression (EX07) ;
- load_model : recharge le modèle et son seuil, refuse un fichier absent ou illisible (EX05) ;
- predict : valide la demande, renvoie les probabilités de défaut et les décisions (EX04) ;
- interpret : niveau de risque et recommandation (EX12, table de restitution du plan de test).

Réentraînement et rapports, depuis la racine du projet : python -m app.prediction
(ajouter --save-baseline pour créer la référence de non-régression).
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.preprocessing import DATA_PATH, SEED, load_data, preprocess_data, split_data
from app.validation import CATEGORIES, FEATURES, INCOME_COLUMN, NUMERIC_FEATURES, validate_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

# Table de restitution : p < seuil / 2 -> faible ; seuil / 2 <= p < seuil -> modéré ; p >= seuil -> élevé
RISK_LEVELS = {
    "faible": "Favorable",
    "modéré": "Étude approfondie du dossier",
    "élevé": "Défavorable (défaut prédit)",
}


# Tolérances de non-régression, fixées AVANT toute comparaison (plan de test, EX07)
METRIC_TOLERANCE = 0.02  # baisse maximale tolérée d'une métrique (en valeur absolue)
PROBA_TOLERANCE = 0.05  # écart maximal toléré sur la probabilité d'une même demande
RELOAD_TOLERANCE = 1e-12  # écart toléré entre avant et après sauvegarde et rechargement
CRITICAL_METRICS = ("recall", "f1")  # baisse > tolérance : version rejetée
WATCHED_METRICS = ("precision", "roc_auc")  # baisse > tolérance : investigation



# ==================================================================================================
# Entraînement et choix du seuil
# ==================================================================================================

# Variables transformées par log(1 + x) avant standardisation : distribution très asymétrique
# (revenu : asymétrie 1,57 -> -0,78). Transformation croissante : l'ordre des valeurs est conservé.
LOG_FEATURES = [INCOME_COLUMN]

MIN_RECALL = 0.75  # exigence sur le test (T12)
VAL_MIN_RECALL = 0.80  # rappel visé sur la validation : marge sur l'exigence (T10)

# Objectif : estimer la probabilité de défaut de remboursement d'un demandeur.
# Modèle retenu : la régression logistique (choix prioritaire) ; probabilités interprétables
# et coefficients explicables à un agent de crédit. Pas de class_weight="balanced" :
# il gonflerait les probabilités (déséquilibre modéré, 33 % de défauts) ; la décision passe par le seuil.
MAIN_MODEL = "logistic_regression"


def _pipeline(model) -> Pipeline:
    """log(1 + x) puis scaler sur le revenu, scaler sur les autres variables numériques,
    one-hot sur les catégorielles, puis le modèle."""
    log_scaler = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    transform = ColumnTransformer([
        ("log", log_scaler, LOG_FEATURES),
        ("num", StandardScaler(), [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]),
        ("cat", OneHotEncoder(categories=list(CATEGORIES.values())), list(CATEGORIES)),
    ])
    return Pipeline([("prep", transform), ("model", model)])


# Le modèle retenu, puis deux modèles de comparaison (servent de repère, pas de candidats)
MODELS = {
    MAIN_MODEL: _pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
    "decision_tree": _pipeline(DecisionTreeClassifier(max_depth=5, random_state=SEED)),
    "random_forest": _pipeline(RandomForestClassifier(n_estimators=200, random_state=SEED)),
}


def train_model(X_train, y_train, model_name=MAIN_MODEL):
    """Entraîne une copie neuve (clone) du pipeline : les transformations ne voient que X_train."""
    if model_name not in MODELS:
        raise ValueError(f"Modèle inconnu : {model_name} (choix : {', '.join(MODELS)})")
    return clone(MODELS[model_name]).fit(X_train, y_train)


def train_and_validate(X_train, y_train, X_val, y_val, model_name=MAIN_MODEL, pipeline=None,
                       min_recall=VAL_MIN_RECALL):
    """Entraîne un modèle (ou un pipeline candidat), choisit son seuil sur la validation.
    Retourne (modèle, seuil, probabilités de validation)."""
    model = clone(pipeline).fit(X_train, y_train) if pipeline is not None else train_model(
        X_train, y_train, model_name)
    proba_val = model.predict_proba(X_val)[:, 1]
    return model, choose_threshold(y_val, proba_val, min_recall), proba_val


def choose_threshold(y_true, proba, min_recall=VAL_MIN_RECALL):
    """Seuil choisi sur la VALIDATION (T15) : parmi les seuils i / 100 qui atteignent le rappel
    visé, celui de meilleure précision. ValueError si aucun seuil n'atteint ce rappel."""
    best_threshold, best_precision = None, -1.0
    for i in range(1, 100):
        threshold = i / 100
        y_pred = (proba >= threshold).astype(int)
        if recall_score(y_true, y_pred, zero_division=0) < min_recall:
            continue
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision >= best_precision:  # à précision égale, le seuil le plus haut
            best_threshold, best_precision = threshold, precision
    if best_threshold is None:
        raise ValueError(f"Aucun seuil n'atteint un rappel de {min_recall}")
    return best_threshold


# ==================================================================================================
# Évaluation et non-régression
# ==================================================================================================

def evaluate_model(y_true, proba, threshold):
    """Métriques au seuil donné (T20) : rappel, précision, F1, PR-AUC, ROC-AUC, FPR, FNR,
    matrice de confusion, qualité des probabilités (Brier, log loss) ; l'accuracy est indicative."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        # FPR : bons payeurs refusés à tort ; FNR : défauts non détectés
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def code_version():
    """Empreinte du code de app/ (SHA-256 des sources) et commit git s'il existe."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "app").glob("*.py")):
        digest.update(path.read_bytes())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "aucun"
    return {"fingerprint": digest.hexdigest()[:12], "commit": commit}


def describe_version(model, name, threshold, metrics, proba_val, data_path):
    """Tout ce qu'il faut pour comparer deux versions : code, modèle, bibliothèques, graine,
    seuil, métriques et probabilités de validation, empreinte des données."""
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "code": code_version(),
        "model": {"name": name, "params": {k: str(v) for k, v in model[-1].get_params().items()}},
        "libraries": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__, "joblib": joblib.__version__},
        "seed": SEED,
        "threshold": threshold,
        "validation_metrics": metrics,
        "validation_proba": [float(p) for p in proba_val],
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
    }


def compare_to_baseline(baseline, current, tolerance=METRIC_TOLERANCE, proba_tolerance=PROBA_TOLERANCE):
    """Compare une version à la référence, sur le même jeu de validation. Retourne (décision, motifs).

    - « rejeter »     : le rappel ou le F1 baisse de plus de la tolérance ;
    - « investiguer » : la précision ou la ROC-AUC baisse de plus de la tolérance, une probabilité
                        varie de plus de la tolérance, ou les données ou bibliothèques ont changé ;
    - « accepter »    : sinon (les variations restent dans les tolérances documentées).
    """
    reject, investigate = [], []
    for group, metrics in ((reject, CRITICAL_METRICS), (investigate, WATCHED_METRICS)):
        for metric in metrics:
            drop = baseline["validation_metrics"][metric] - current["validation_metrics"][metric]
            if drop > tolerance:
                group.append(f"{metric} en baisse de {drop:.4f} (tolérance {tolerance})")
    if baseline["data_sha256"] != current["data_sha256"]:
        investigate.append("jeu de données différent de la référence : probabilités non comparables")
    else:
        gap = float(np.max(np.abs(np.array(baseline["validation_proba"]) - current["validation_proba"])))
        if gap > proba_tolerance:
            investigate.append(f"probabilité modifiée de {gap:.4f} au maximum (tolérance {proba_tolerance})")
    if baseline["libraries"] != current["libraries"]:
        investigate.append("versions des bibliothèques différentes de la référence")
    if reject:
        return "rejeter", reject + investigate
    if investigate:
        return "investiguer", investigate
    return "accepter", ["aucune variation au-delà des tolérances"]


# ==================================================================================================
# Sauvegarde et chargement
# ==================================================================================================

def save_model(model, threshold, path, reference=None, model_name=MAIN_MODEL):
    """Sauvegarde le modèle, son seuil et ses métadonnées (joblib).
    reference : données d'entraînement servant aux alertes (check_input)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "threshold": threshold,
        "model_name": model_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "reference": reference,
    }, path)


def sha256_of(path):
    """Empreinte SHA-256 d'un fichier (identifiant de version du modèle affiché par l'API)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(path):
    """Recharge le modèle et son seuil ; retourne (model, threshold)."""
    saved = load_bundle(path)
    return saved["model"], saved["threshold"]


def load_bundle(path):
    """Recharge le contenu complet du fichier modèle (modèle, seuil, métadonnées).

    FileNotFoundError si le fichier est absent ; ValueError si le contenu est illisible ou
    incomplet. joblib.load pouvant exécuter du code contenu dans le fichier, seul un modèle
    produit par ce projet doit être chargé.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle indisponible : fichier introuvable ({path})")
    try:
        saved = joblib.load(path)
    except Exception as error:
        raise ValueError(f"Fichier modèle illisible : {path}") from error
    if not isinstance(saved, dict) or "model" not in saved or "threshold" not in saved:
        raise ValueError(f"Fichier modèle invalide (modèle ou seuil manquant) : {path}")
    return saved


# ==================================================================================================
# Prédiction et restitution
# ==================================================================================================

def predict(model, X, threshold=0.5):
    """Valide les demandes puis retourne (probabilités de défaut, décisions 0/1 au seuil)."""
    validate_data(X, require_target=False)
    proba = model.predict_proba(X[FEATURES])[:, 1]
    return proba, (proba >= threshold).astype(int)


def interpret(proba, threshold):
    """Restitution pour l'agent de crédit : une liste de dictionnaires, un par demande."""
    results = []
    for p in proba:
        level = "élevé" if p >= threshold else "modéré" if p >= threshold / 2 else "faible"
        results.append({
            "probabilite_defaut": float(p),
            "decision": int(p >= threshold),
            "niveau_risque": level,
            "recommandation": RISK_LEVELS[level],
        })
    return results


if __name__ == "__main__":
    # 1. Données : validation, doublons supprimés, découpage stratifié 60/20/20
    X, y = preprocess_data(load_data(DATA_PATH))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train {len(X_train)} | Validation {len(X_val)} | Test {len(X_test)}")

    # 2. Entraînement et seuil choisi sur la VALIDATION ; les deux autres modèles servent de repère
    validation, models, probas = {}, {}, {}
    for name in MODELS:
        models[name], threshold, probas[name] = train_and_validate(X_train, y_train, X_val, y_val, name)
        validation[name] = {"threshold": threshold, **evaluate_model(y_val, probas[name], threshold)}
    threshold = validation[MAIN_MODEL]["threshold"]

    # 3. Sauvegarde du modèle retenu, puis rechargement
    save_model(models[MAIN_MODEL], threshold, MODEL_PATH, reference=X_train)
    model, loaded_threshold = load_model(MODEL_PATH)

    # 4. Évaluation UNIQUE sur le test, avec le modèle rechargé ; train pour mesurer le surapprentissage
    proba_test = model.predict_proba(X_test)[:, 1]
    test = evaluate_model(y_test, proba_test, loaded_threshold)
    train = evaluate_model(y_train, model.predict_proba(X_train)[:, 1], loaded_threshold)
    gap = {k: round(train[k] - test[k], 4) for k in ("recall", "precision", "roc_auc", "brier")}
    max_diff = float(abs(models[MAIN_MODEL].predict_proba(X_test)[:, 1] - proba_test).max())

    # 5. Rapports
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    pd.DataFrame(validation).T.round(4).to_csv(reports / "comparaison_modeles.csv")
    (reports / "metrics.json").write_text(json.dumps({
        "seed": SEED, "model": MAIN_MODEL, "threshold": threshold, "validation": validation,
        "test": test, "train": train, "ecart_train_test": gap}, indent=2))
    (reports / "reproductibilite.json").write_text(json.dumps({
        "model_file": "models/model.pkl", "n_predictions": len(proba_test), "max_abs_diff": max_diff,
        "tolerance": RELOAD_TOLERANCE, "threshold_saved": threshold, "threshold_loaded": loaded_threshold,
        "reproducible": max_diff <= RELOAD_TOLERANCE and loaded_threshold == threshold}, indent=2))

    columns = ["threshold", "recall", "precision", "roc_auc", "pr_auc", "brier"]
    print(pd.DataFrame(validation).T[columns].round(3))
    print(f"\nModèle retenu : {MAIN_MODEL}, seuil {threshold} -> {MODEL_PATH.relative_to(ROOT)}")
    print("Test :", {k: round(test[k], 3) for k in ("recall", "precision", "roc_auc", "brier", "fpr")})
    print("Écart train - test :", gap)
    print(f"Reproductibilité : écart maximal {max_diff} (tolérance {RELOAD_TOLERANCE})")

    # 6. Non-régression : description de la version courante, référence figée, comparaison
    version = describe_version(models[MAIN_MODEL], MAIN_MODEL, threshold, validation[MAIN_MODEL],
                               probas[MAIN_MODEL], DATA_PATH)
    (reports / "version_courante.json").write_text(json.dumps(version, indent=2))
    baseline_path = reports / "baseline.json"
    if "--save-baseline" in sys.argv:  # référence créée sur demande explicite, jamais écrasée
        if baseline_path.exists():
            print("Référence NON modifiée : reports/baseline.json existe déjà")
        else:
            baseline_path.write_text(json.dumps(version, indent=2))
            print(f"Référence enregistrée : reports/{baseline_path.name}")
    if baseline_path.exists():
        decision, reasons = compare_to_baseline(json.loads(baseline_path.read_text()), version)
        (reports / "non_regression.json").write_text(json.dumps(
            {"decision": decision, "motifs": reasons}, indent=2, ensure_ascii=False))
        print(f"Non-régression : {decision.upper()} ({'; '.join(reasons)})")
