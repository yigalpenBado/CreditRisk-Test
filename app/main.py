"""API de prédiction du risque de défaut de remboursement.

Endpoints (contrat documenté dans docs/plan_de_test.md et docs/resultats_tests_api.md) :
- GET  /health     : disponibilité du service et du modèle (200, ou 503 sans modèle) ;
- POST /predict    : estimation du risque d'UNE demande (200, 422 si invalide, 503 sans modèle) ;
- GET  /model-info : métadonnées de la version du modèle (200, ou 503 sans modèle).

Validation de la demande par Pydantic, avant le modèle (réponse 422) : champ manquant,
champ supplémentaire, type incorrect (pas de "35" pour 35 ni de 12.5 pour une durée),
valeur hors intervalle, modalité inconnue, valeur non finie, JSON mal formé.

Lancement depuis la racine du projet : uvicorn app.main:app --port 8001 --workers 2
(2 processus : objectif de latence sous charge O3 ; chacun charge son propre exemplaire du modèle).
Modèle chargé depuis models/model.pkl, ou depuis la variable d'environnement MODEL_PATH.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.prediction import RISK_LEVELS, interpret, load_bundle, sha256_of
from app.validation import (
    AGE_RANGE,
    CATEGORIES,
    DURATION_RANGE,
    FEATURES,
    INCOME_COLUMN,
    check_input,
    validate_data,
)

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("api_credit")
# Messages renvoyés au client : génériques, sans trace technique (chemin, exception, bibliothèque)
UNAVAILABLE = "Modèle indisponible : le modèle n'est pas chargé"
INTERNAL_ERROR = "Erreur interne : la prédiction n'a pas pu être réalisée"

# Schéma d'une demande : les 21 variables du modèle, aucun champ supplémentaire, types stricts
fields = {
    "Age": (int, Field(ge=AGE_RANGE[0], le=AGE_RANGE[1], description="Âge en années")),
    "LoanDuration": (int, Field(ge=DURATION_RANGE[0], le=DURATION_RANGE[1], description="Durée en mois")),
    "LoanAmount": (float, Field(gt=0, allow_inf_nan=False, description="Montant demandé")),
    INCOME_COLUMN: (float, Field(gt=0, allow_inf_nan=False, description="Revenu mensuel")),
    "InstallmentPercent": (int, Field(ge=1, le=6, description="Code de part du revenu (1 à 6)")),
    "CurrentResidenceDuration": (int, Field(ge=1)),
    "ExistingCreditsCount": (int, Field(ge=1)),
    "Dependents": (int, Field(ge=1)),
}
for col, values in CATEGORIES.items():
    fields[col] = (Literal[tuple(values)], Field())
Demande = create_model(
    "Demande",
    __config__=ConfigDict(extra="forbid", strict=True),
    **{col: fields[col] for col in FEATURES},
)


class Reponse(BaseModel):
    probabilite_defaut: float
    decision: int
    niveau_risque: str
    recommandation: str
    seuil: float
    alertes: list[str]
    version_modele: str


# Modèle chargé une fois (au démarrage, ou au premier appel si le démarrage n'a pas eu lieu)
state = {"loaded": False, "bundle": None, "version": None}


def reload_model():
    """(Re)charge le modèle ; en cas d'échec, le service reste joignable mais signale 503."""
    path = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.pkl"))
    state.update(loaded=True, bundle=None, version=None)
    try:
        bundle = load_bundle(path)
        # Le modèle doit attendre exactement les variables du contrat, dans le même ordre :
        # sinon les dimensions ne correspondent pas et chaque prédiction échouerait
        expected = list(getattr(bundle["model"], "feature_names_in_", []))
        if expected != FEATURES:
            raise ValueError(f"Variables du modèle incompatibles avec le contrat : {expected}")
    except (FileNotFoundError, ValueError) as error:
        logger.error("Chargement du modèle impossible : %s", error)  # détail côté serveur uniquement
        return
    state.update(bundle=bundle, version=sha256_of(path)[:12])


def get_bundle():
    if not state["loaded"]:
        reload_model()
    if state["bundle"] is None:
        raise HTTPException(status_code=503, detail=UNAVAILABLE)
    return state["bundle"]


@asynccontextmanager
async def lifespan(app):
    reload_model()
    yield


app = FastAPI(title="Risque de défaut de remboursement", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, error):
    """Demande invalide : 422 avec le type d'erreur, le champ et le message. La valeur reçue n'est
    pas renvoyée : elle peut être non sérialisable (NaN, infini) ou contenir une donnée personnelle."""
    detail = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in error.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(Exception)
async def unexpected_error(request, error):
    """Dernier filet : toute erreur non prévue donne une 500 générique, la trace reste côté serveur."""
    logger.exception("Erreur non gérée sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": INTERNAL_ERROR})


@app.get("/health")
def health():
    """Disponibilité du service : 200 si le modèle est chargé, 503 sinon."""
    try:
        get_bundle()
    except HTTPException:
        return JSONResponse(status_code=503, content={"status": "indisponible", "modele_charge": False})
    return {"status": "ok", "modele_charge": True}


@app.get("/model-info")
def model_info():
    """Métadonnées de la version du modèle."""
    bundle = get_bundle()
    return {
        "modele": bundle.get("model_name") or type(bundle["model"]).__name__,
        "version": state["version"],
        "date_entrainement": bundle.get("trained_at"),
        "seuil": bundle["threshold"],
        "variables": list(bundle["model"].feature_names_in_),  # lues dans le modèle chargé
    }


@app.post("/predict", response_model=Reponse)
def predict_risk(demande: Demande):
    """Estimation du risque de défaut d'une demande de crédit."""
    bundle = get_bundle()
    X = pd.DataFrame([demande.model_dump()])[FEATURES]
    # 1. Validation métier (seconde ligne de défense après Pydantic) : erreur du client -> 422,
    #    au même format que les erreurs de schéma : liste de {"type", "loc", "msg"}
    try:
        validate_data(X, require_target=False)
    except ValueError as error:
        detail = [{"type": "value_error", "loc": ["body"], "msg": str(error)}]
        raise HTTPException(status_code=422, detail=detail) from error
    # 2. Prétraitement et prédiction : une erreur ici n'est pas due au client -> 500 générique,
    #    le détail (exception, trace) n'est écrit que dans le journal du serveur
    try:
        proba = bundle["model"].predict_proba(X)[:, 1]
        if proba.shape != (1,) or not 0 <= proba[0] <= 1:
            raise ValueError(f"Sortie du modèle invalide : {proba!r}")
    except Exception:
        logger.exception("Échec de la prédiction")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR) from None
    result = interpret(proba, bundle["threshold"])[0]
    reference = bundle.get("reference")
    alerts = check_input(reference, X) if reference is not None else []
    # Demande inhabituelle (hors plage d'entraînement, incohérente) : la probabilité est peu fiable,
    # une recommandation favorable est remplacée par une étude approfondie du dossier
    if alerts and result["niveau_risque"] == "faible":
        result["recommandation"] = RISK_LEVELS["modéré"]
    # Journal : la décision uniquement, aucune donnée personnelle de la demande
    logger.info("Prédiction : décision=%d, niveau=%s", result["decision"], result["niveau_risque"])
    return {**result, "seuil": bundle["threshold"], "alertes": alerts, "version_modele": state["version"]}
