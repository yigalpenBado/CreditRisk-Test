"""Tests de robustesse (Partie IV) : entrées limites et comportements inattendus.

Deux comportements attendus, à ne pas confondre :
- entrée INVALIDE -> rejet 422 (valeur manquante, non numérique, non finie, hors intervalle,
  schéma incorrect) : le modèle n'est pas appelé ;
- entrée INHABITUELLE mais VALIDE -> 200 avec alerte (aux limites, éloignée des données
  d'entraînement) : la prédiction respecte son contrat et l'agent est averti.
Le registre des anomalies est dans docs/registre_anomalies.md.
Les tests de prédiction, de restitution et de chargement sécurisé du modèle (T16 à T19) sont en fin
de fichier.
"""

import json
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Fixtures partagées, définies dans test_validation.py (pytest les résout par leur nom)
from test_validation import DEMANDE, client, model, model_path, train_data, valid_df  # noqa: F401

from app.prediction import interpret, load_model, predict, save_model
from app.validation import CATEGORIES, INCOME_COLUMN, expected_income


def envoyer(client, **changes):
    return client.post("/predict", json={**DEMANDE, **changes})


def types_erreur(response):
    assert response.status_code == 422
    return {e["type"] for e in response.json()["detail"]}


def assert_contrat(response):
    """Contrat de sortie d'une prédiction : 200, probabilité finie dans [0, 1], décision 0/1."""
    assert response.status_code == 200
    resultat = response.json()
    assert math.isfinite(resultat["probabilite_defaut"])
    assert 0.0 <= resultat["probabilite_defaut"] <= 1.0
    assert resultat["decision"] in (0, 1)
    return resultat


# 1. Valeurs manquantes ou non numériques : rejet, sans conversion implicite (ni null, ni "35")
@pytest.mark.parametrize("champ, valeur", [("Age", None), ("Age", "35"), ("LoanAmount", "abc")])
def test_valeur_invalide_rejetee(client, champ, valeur):
    assert types_erreur(envoyer(client, **{champ: valeur})) <= {"int_type", "float_type"}


# 2. Valeurs non finies (NaN, Infinity) : rejet 422, et non erreur 500 (anomalie A8)
@pytest.mark.parametrize("litteral", ["NaN", "Infinity"])
def test_valeur_non_finie_rejetee(client, litteral):
    corps = json.dumps(DEMANDE).replace('"LoanAmount": 1000.0', f'"LoanAmount": {litteral}')
    response = client.post("/predict", content=corps, headers={"Content-Type": "application/json"})
    assert types_erreur(response) == {"finite_number"}


# 3. Limites des intervalles : la borne est acceptée, la valeur juste au-delà est rejetée
@pytest.mark.parametrize("champ, borne, au_dela", [("Age", 75, 76), ("LoanDuration", 72, 73)])
def test_limites_des_intervalles(client, champ, borne, au_dela):
    assert_contrat(envoyer(client, **{champ: borne}))
    assert types_erreur(envoyer(client, **{champ: au_dela})) == {"less_than_equal"}


# 4. Entrées très éloignées des données d'entraînement : traitées, signalées par une alerte et
#    jamais recommandées favorablement (anomalie A9)
@pytest.mark.parametrize("changes", [{"LoanAmount": 1e300}, {INCOME_COLUMN: 1e-300}])
def test_entree_eloignee_du_train(client, changes):
    resultat = assert_contrat(envoyer(client, **changes))
    assert any("hors de la plage d'entraînement" in a for a in resultat["alertes"])
    assert resultat["recommandation"] != "Favorable"


# 5. Schéma incorrect : rejet 422 (noms de champs en minuscules, objet vide, liste au lieu d'un objet)
def test_schema_incorrect_rejete(client):
    for corps in ({k.lower(): v for k, v in DEMANDE.items()}, {}, [DEMANDE]):
        assert client.post("/predict", json=corps).status_code == 422


# Contrat de sortie sur 50 demandes valides générées aléatoirement, valeurs extrêmes comprises
demandes_valides = st.fixed_dictionaries({
    **{col: st.sampled_from(values) for col, values in CATEGORIES.items()},
    "Age": st.integers(18, 75), "LoanDuration": st.integers(1, 72), "InstallmentPercent": st.integers(1, 6),
    "CurrentResidenceDuration": st.integers(1, 10), "ExistingCreditsCount": st.integers(1, 10),
    "Dependents": st.integers(1, 10),
    "LoanAmount": st.floats(0.01, 1e15), INCOME_COLUMN: st.floats(0.01, 1e15),
})


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(demande=demandes_valides)
def test_contrat_de_sortie_sur_entrees_aleatoires(client, demande):
    assert_contrat(client.post("/predict", json=demande))


# Stabilité : de faibles variations (âge ± 1 an, durée ± 1 mois, montant ± 1 %, revenu recalculé)
# changent au plus 5 % des décisions
def test_stabilite_aux_petites_variations(model, train_data):
    X = train_data[0]
    reference = model.predict_proba(X)[:, 1] >= 0.4
    rng = np.random.default_rng(0)
    variee = X.assign(
        Age=(X["Age"] + rng.choice([-1, 1], len(X))).clip(18, 75),
        LoanDuration=(X["LoanDuration"] + rng.choice([-1, 1], len(X))).clip(1, 72),
        LoanAmount=X["LoanAmount"] * rng.choice([0.99, 1.01], len(X)),
    )
    variee[INCOME_COLUMN] = expected_income(variee["LoanAmount"], variee["LoanDuration"],
                                            variee["InstallmentPercent"]).round(2)
    changements = (model.predict_proba(variee)[:, 1] >= 0.4) != reference
    assert changements.mean() <= 0.05


# --- Prédiction, restitution et chargement sécurisé du modèle ---

# T16, T17 : probabilités valides, décision au seuil, niveau de risque et recommandation
def test_prediction_et_restitution(model, valid_df):
    proba, decisions = predict(model, valid_df, threshold=0.4)
    assert len(proba) == 20 and ((proba >= 0) & (proba <= 1)).all()
    assert decisions.tolist() == (proba >= 0.4).astype(int).tolist()

    resultats = interpret([0.0, 0.2, 0.4], threshold=0.4)  # valeurs limites : seuil / 2 et seuil
    assert [r["niveau_risque"] for r in resultats] == ["faible", "modéré", "élevé"]
    assert [r["recommandation"] for r in resultats] == [
        "Favorable", "Étude approfondie du dossier", "Défavorable (défaut prédit)"]


# T18, T19 : modèle rechargé à l'identique ; absent ou illisible : refusé
def test_chargement_du_modele(model, valid_df, tmp_path):
    path = tmp_path / "model.pkl"
    save_model(model, 0.4, path)
    loaded, threshold = load_model(path)
    assert threshold == 0.4
    assert np.array_equal(loaded.predict_proba(valid_df), model.predict_proba(valid_df))

    with pytest.raises(FileNotFoundError, match="Modèle indisponible"):
        load_model(tmp_path / "absent.pkl")
    path.write_bytes(b"contenu corrompu")
    with pytest.raises(ValueError, match="Fichier modèle illisible"):
        load_model(path)
