"""Tests d'intégration (Partie III) : validation, prétraitement, modèle et API ensemble.

Le fixture `pipeline` exécute une fois la chaîne complète sur un jeu synthétique écrit en CSV :
load_data -> preprocess_data -> split_data -> train_model -> choose_threshold -> save_model -> API.
Dans les scénarios d'erreur, la réponse ne doit contenir aucune trace technique.
"""

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Fixtures partagées, définies dans test_validation.py (pytest les résout par leur nom)
from test_validation import train_data  # noqa: F401

import app.main as api
from app.prediction import choose_threshold, load_model, save_model, sha256_of, train_model
from app.preprocessing import load_data, preprocess_data, split_data
from app.validation import FEATURES, ID_COLUMN, INCOME_COLUMN, TARGET

TOLERANCE = 1e-12
TRACES = ["Traceback", 'File "', "/home", ".py", "sklearn", "LogisticRegression", "Error", "features"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, train_data):
    folder = tmp_path_factory.mktemp("integration")
    X, y = train_data
    raw = X.assign(**{TARGET: y.map({0: "No Risk", 1: "Risk"}), ID_COLUMN: range(len(X))})
    raw.to_csv(folder / "demandes.csv", index=False)

    X, y = preprocess_data(load_data(folder / "demandes.csv"))
    X_train, X_val, X_test, y_train, y_val, _ = split_data(X, y)
    model = train_model(X_train, y_train)
    threshold = choose_threshold(y_val, model.predict_proba(X_val)[:, 1])
    save_model(model, threshold, folder / "model.pkl", reference=X_train)
    return {"X_train": X_train, "X_test": X_test, "model": model, "threshold": threshold,
            "path": folder / "model.pkl"}


@pytest.fixture
def client(pipeline, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(pipeline["path"]))
    api.reload_model()
    return TestClient(api.app)


@pytest.fixture
def demande(pipeline):
    row = pipeline["X_test"].iloc[0]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def model_charge():
    return api.state["bundle"]["model"]


def assert_erreur_propre(response, code, detail):
    assert response.status_code == code
    assert response.json() == {"detail": detail}
    assert not any(t in response.text for t in TRACES)


# Scénario principal : requête valide -> validation -> transformation -> prédiction
# -> structure de la réponse -> métadonnées cohérentes
def test_scenario_principal(client, demande, pipeline, monkeypatch):
    recu = []
    predict_proba = model_charge().predict_proba
    monkeypatch.setattr(model_charge(), "predict_proba", lambda X: recu.append(X) or predict_proba(X))

    response = client.post("/predict", json=demande)
    assert response.status_code == 200

    # Validation : une demande invalide est arrêtée avant le modèle
    assert client.post("/predict", json={**demande, "Age": 17}).status_code == 422
    assert len(recu) == 1

    # Transformation : 21 variables dans l'ordre du contrat -> 61 colonnes finies,
    # revenu standardisé avec les statistiques du train
    assert list(recu[0].columns) == FEATURES
    prep = model_charge().named_steps["prep"]
    Z = prep.transform(recu[0])
    assert Z.shape == (1, model_charge().named_steps["model"].n_features_in_) and np.isfinite(Z).all()
    log_income = np.log1p(pipeline["X_train"][INCOME_COLUMN])
    attendu = (np.log1p(demande[INCOME_COLUMN]) - log_income.mean()) / log_income.std(ddof=0)
    assert Z[0, 0] == pytest.approx(attendu)

    # Prédiction : probabilité identique à celle du modèle entraîné, décision au seuil
    r = response.json()
    attendu = pipeline["model"].predict_proba(recu[0])[0, 1]
    assert abs(r["probabilite_defaut"] - attendu) <= TOLERANCE
    assert r["decision"] == int(attendu >= pipeline["threshold"])

    # Structure de la réponse
    assert set(r) == {"probabilite_defaut", "decision", "niveau_risque", "recommandation",
                      "seuil", "alertes", "version_modele"}

    # Métadonnées : version = empreinte du fichier, seuil et variables du modèle chargé
    info = client.get("/model-info").json()
    assert info["version"] == sha256_of(pipeline["path"])[:12] == r["version_modele"]
    assert info["seuil"] == pipeline["threshold"] == r["seuil"]
    assert info["variables"] == FEATURES


# Reproductibilité : mêmes probabilités avant sauvegarde et après rechargement ; même seuil
def test_reproductibilite(pipeline):
    model, threshold = load_model(pipeline["path"])
    avant = pipeline["model"].predict_proba(pipeline["X_test"])[:, 1]
    apres = model.predict_proba(pipeline["X_test"])[:, 1]
    assert threshold == pipeline["threshold"] and np.abs(avant - apres).max() <= TOLERANCE


# Modèle illisible : 503, message générique, sans trace technique (EX05, EX16)
def test_modele_illisible(demande, tmp_path, monkeypatch):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"contenu corrompu")
    monkeypatch.setenv("MODEL_PATH", str(path))
    api.reload_model()
    client = TestClient(api.app)
    assert client.get("/health").status_code == 503
    assert_erreur_propre(client.post("/predict", json=demande), 503, api.UNAVAILABLE)


# Variable attendue absente : erreur explicite dans la chaîne (fichier), 422 dans l'API (requête)
def test_variable_absente(client, demande, pipeline, tmp_path):
    incomplet = pipeline["X_train"].drop(columns=INCOME_COLUMN).assign(Risk="Risk")
    incomplet.to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="Colonne manquante : MonthlyIncome"):
        preprocess_data(load_data(tmp_path / "x.csv"))
    response = client.post("/predict", json={k: v for k, v in demande.items() if k != INCOME_COLUMN})
    assert response.status_code == 422 and response.json()["detail"][0]["loc"] == ["body", INCOME_COLUMN]


# Le prétraitement retourne une erreur : 500 générique, sans trace technique
def test_erreur_du_pretraitement(client, demande, monkeypatch):
    def echec(X):
        raise ValueError("Found unknown categories in column 3")

    monkeypatch.setattr(model_charge().named_steps["prep"], "transform", echec)
    assert_erreur_propre(client.post("/predict", json=demande), 500, api.INTERNAL_ERROR)


# Dimensions incompatibles avec le modèle (60 colonnes attendues au lieu de 61) : 500 générique
def test_dimensions_incompatibles(client, demande, monkeypatch):
    estimator = model_charge().named_steps["model"]
    monkeypatch.setattr(estimator, "coef_", estimator.coef_[:, :-1])
    monkeypatch.setattr(estimator, "n_features_in_", estimator.n_features_in_ - 1)
    assert_erreur_propre(client.post("/predict", json=demande), 500, api.INTERNAL_ERROR)


# Erreur interne pendant la prédiction : 500 générique, détail dans le journal ; service toujours
# disponible ensuite (EX05, EX16)
def test_erreur_interne(client, demande, monkeypatch, caplog):
    def panne(X):
        raise RuntimeError("connexion perdue vers /home/user/secret")

    monkeypatch.setattr(model_charge(), "predict_proba", panne)
    with caplog.at_level(logging.ERROR, logger="api_credit"):
        assert_erreur_propre(client.post("/predict", json=demande), 500, api.INTERNAL_ERROR)
    assert "connexion perdue" in caplog.text
    assert client.get("/health").status_code == 200
