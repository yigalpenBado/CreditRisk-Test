"""Tests fonctionnels de l'API (Partie II) : GET /health, POST /predict, GET /model-info.

Un test par scénario obligatoire (S1 à S10). Erreur de validation (422) :
{"detail": [{"type", "loc", "msg"}]}. Le modèle est entraîné sur le jeu synthétique
(fixtures `client` et `model_path` de tests/test_validation.py).
"""

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import fastapi
import httpx
import numpy as np
import pandas as pd
import psutil
import pytest
import sklearn
from fastapi.testclient import TestClient

# Fixtures partagées, définies dans test_validation.py (pytest les résout par leur nom)
from test_validation import DEMANDE, client, model, model_path, train_data  # noqa: F401

from app.main import UNAVAILABLE, app, reload_model
from app.prediction import interpret, load_model
from app.preprocessing import DATA_PATH, load_data, preprocess_data, split_data
from app.validation import FEATURES, check_input, validate_data


def post(client, **changes):
    return client.post("/predict", json={**DEMANDE, **changes})


def erreurs(response):
    """(type, champ) de chaque erreur de validation."""
    return [(e["type"], e["loc"][-1]) for e in response.json()["detail"]]


# /health et /model-info : 200, service disponible et métadonnées du modèle
def test_health_et_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "modele_charge": True}
    info = client.get("/model-info").json()
    assert info["modele"] == "logistic_regression" and info["seuil"] == 0.4
    assert info["variables"] == FEATURES and len(info["version"]) == 12


# S1. Requête valide : 200, probabilité dans [0, 1], décision cohérente avec le seuil, aucune alerte
def test_requete_valide(client):
    response = post(client)
    assert response.status_code == 200
    r = response.json()
    assert 0 <= r["probabilite_defaut"] <= 1
    assert r["decision"] == int(r["probabilite_defaut"] >= r["seuil"])
    assert r["niveau_risque"] in {"faible", "modéré", "élevé"} and r["alertes"] == []


# S2. Champ obligatoire absent : 422 « missing » sur ce champ
def test_champ_obligatoire_absent(client):
    response = client.post("/predict", json={k: v for k, v in DEMANDE.items() if k != "Age"})
    assert response.status_code == 422
    assert erreurs(response) == [("missing", "Age")]


# S3. Âge négatif : 422 « greater_than_equal » sur Age
def test_age_negatif(client):
    assert erreurs(post(client, Age=-5)) == [("greater_than_equal", "Age")]


# S4. Revenu sous forme de texte : 422 « float_type » (pas de conversion implicite)
def test_revenu_texte(client):
    assert erreurs(post(client, MonthlyIncome="250000")) == [("float_type", "MonthlyIncome")]


# S5. Montant nul ou négatif : 422 « greater_than » sur LoanAmount
def test_montant_nul_ou_negatif(client):
    assert erreurs(post(client, LoanAmount=0)) == [("greater_than", "LoanAmount")]
    assert erreurs(post(client, LoanAmount=-1000)) == [("greater_than", "LoanAmount")]


# S6. Durée non entière : 422 « int_type » sur LoanDuration
def test_duree_non_entiere(client):
    assert erreurs(post(client, LoanDuration=12.5)) == [("int_type", "LoanDuration")]


# S7. Champ supplémentaire : 422 « extra_forbidden »
def test_champ_supplementaire(client):
    assert erreurs(post(client, revenu_mensuel=1)) == [("extra_forbidden", "revenu_mensuel")]


# S8. Valeurs extrêmement élevées : montant et revenu traités (200) avec alertes ; âge borné (422)
def test_valeurs_extremement_elevees(client):
    response = post(client, LoanAmount=1e12, MonthlyIncome=1e12)
    assert response.status_code == 200
    assert 0 <= response.json()["probabilite_defaut"] <= 1
    assert any("LoanAmount hors de la plage" in a for a in response.json()["alertes"])
    assert erreurs(post(client, Age=10**6)) == [("less_than_equal", "Age")]


# S9. Modèle indisponible : 503 sur les trois endpoints, message sans détail technique
def test_modele_indisponible(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.pkl"))
    reload_model()
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    assert post(client).json() == {"detail": UNAVAILABLE}
    assert client.get("/model-info").status_code == 503


# S10. Requête mal formée : 422 (JSON invalide, liste au lieu d'un objet, corps vide)
def test_requete_mal_formee(client):
    response = client.post("/predict", content=b'{"Age": 35,', headers={"Content-Type": "application/json"})
    assert erreurs(response)[0][0] == "json_invalid"
    assert client.post("/predict", json=[DEMANDE]).status_code == 422
    assert erreurs(client.post("/predict")) == [("missing", "body")]


# --- Performance (Partie VI) ---
# Objectifs fixés AVANT les mesures (usage : un agent saisit une demande et attend la réponse) :
# O1 : latence de l'API, P95 < 100 ms ; O2 : modèle seul, P95 < 10 ms ; O3 : débit >= 20 req/s avec
# P95 < 500 ms sous 16 utilisateurs simultanés (50 agents, une demande par minute : marge x 20) ;
# O4 : mémoire du service < 500 Mo.
# Déploiement : 2 processus uvicorn (O3 n'est pas tenu de façon stable avec un seul processus)
WORKERS = 2
OBJECTIFS = {"api_p95_ms": 100, "modele_p95_ms": 10, "debit_min_rps": 20, "charge_p95_ms": 500,
             "memoire_max_mo": 500}


def stats(times_s):
    ms = np.array(times_s) * 1000
    result = {"moyenne_ms": round(float(ms.mean()), 3)}
    for q in (50, 95, 99):
        result[f"p{q}_ms"] = round(float(np.percentile(ms, q)), 3)
    return {**result, "max_ms": round(float(ms.max()), 3)}


def mesurer(fonction, n=1000, warmup=50):
    """Latence de `fonction` : `warmup` appels d'échauffement, puis n appels chronométrés."""
    for _ in range(warmup):
        fonction()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fonction()
        times.append(time.perf_counter() - start)
    return stats(times)


# O1, vérification rapide exécutée par la CI : P95 de POST /predict < 100 ms (100 requêtes)
def test_latence_p95(client):
    latence = mesurer(lambda: post(client).raise_for_status(), n=100, warmup=10)
    assert latence["p95_ms"] < OBJECTIFS["api_p95_ms"], latence


# Mesures complètes (pytest -m mesures) : modèle réel, 1 000 mesures par niveau (modèle seul,
# prédiction complète, requête HTTP sur un serveur uvicorn réel à WORKERS processus, port 8001), débit
# par taille de lot, charge progressive de 1 à 32 utilisateurs (threads, 100 requêtes chacun), mémoire
# totale du serveur (tous ses processus).
# Résultats : reports/performance.json et reports/performance_charge.png
@pytest.mark.mesures
@pytest.mark.skipif(not DATA_PATH.exists(), reason="jeu de données réel absent")
def test_performance_complete():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, port = DATA_PATH.parents[2], 8001
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                if line.startswith("model name")), platform.processor())
    config = {"systeme": f"{platform.system()} {platform.release()}", "processeur": cpu,
              "coeurs_logiques": os.cpu_count(), "memoire_go": round(psutil.virtual_memory().total / 1e9, 1),
              "python": platform.python_version(), "scikit-learn": sklearn.__version__,
              "pandas": pd.__version__, "fastapi": fastapi.__version__,
              "serveur": f"uvicorn, {WORKERS} processus", "client": "httpx (même machine)"}
    model, threshold = load_model(root / "models" / "model.pkl")
    X, y = preprocess_data(load_data())
    X_train, _, X_test, _, _, _ = split_data(X, y)
    une_demande = pd.DataFrame([DEMANDE])[FEATURES]

    # 1. Modèle seul ; 2. prédiction complète côté application (validation, modèle, restitution, alertes)
    modele = mesurer(lambda: model.predict_proba(une_demande))

    def prediction_complete():
        validate_data(une_demande, require_target=False)
        interpret(model.predict_proba(une_demande)[:, 1], threshold)
        check_input(X_train, une_demande)

    application = mesurer(prediction_complete)

    # 3. Débit du modèle selon la taille du lot
    debit_lots = {}
    for taille in (1, 10, 100, 1000):
        lot = X_test.sample(taille, replace=True, random_state=0)
        moyenne_s = mesurer(lambda: model.predict_proba(lot), n=200, warmup=10)["moyenne_ms"] / 1000
        debit_lots[taille] = round(taille / moyenne_s)

    # 4. API réelle (uvicorn), requêtes séquentielles ; 5. charge progressive ; 6. mémoire du serveur
    serveur = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port),
                                "--workers", str(WORKERS), "--log-level", "warning"], cwd=root)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        with httpx.Client(base_url=url) as http:
            api = mesurer(lambda: http.post("/predict", json=DEMANDE).raise_for_status())

        def utilisateur(_, n_requetes=100):
            times, erreurs = [], 0
            with httpx.Client(base_url=url) as http:
                for _ in range(n_requetes):
                    start = time.perf_counter()
                    erreurs += http.post("/predict", json=DEMANDE).status_code != 200
                    times.append(time.perf_counter() - start)
            return times, erreurs

        charge = []
        for n_users in (1, 2, 4, 8, 16, 32):
            start = time.perf_counter()
            with ThreadPoolExecutor(n_users) as pool:
                resultats = list(pool.map(utilisateur, range(n_users)))
            duree = time.perf_counter() - start
            times = [t for r in resultats for t in r[0]]
            charge.append({"utilisateurs": n_users, "requetes": len(times),
                           "erreurs": sum(r[1] for r in resultats),
                           "debit_rps": round(len(times) / duree, 1), **stats(times)})
        processus = [psutil.Process(serveur.pid), *psutil.Process(serveur.pid).children(recursive=True)]
        memoire_mo = round(sum(p.memory_info().rss for p in processus) / 1e6, 1)
    finally:
        serveur.terminate()
        serveur.wait()

    charge_16 = next(c for c in charge if c["utilisateurs"] == 16)
    verdicts = {
        "O1_api_p95": bool(api["p95_ms"] < OBJECTIFS["api_p95_ms"]),
        "O2_modele_p95": bool(modele["p95_ms"] < OBJECTIFS["modele_p95_ms"]),
        "O3_debit_sous_charge": bool(charge_16["debit_rps"] >= OBJECTIFS["debit_min_rps"]
                                     and charge_16["p95_ms"] < OBJECTIFS["charge_p95_ms"]),
        "O4_memoire": bool(memoire_mo < OBJECTIFS["memoire_max_mo"]),
    }
    (root / "reports" / "performance.json").write_text(json.dumps({
        "configuration": config, "objectifs": OBJECTIFS, "protocole": {"echauffement": 50, "mesures": 1000},
        "modele_seul": modele, "prediction_complete": application, "api_http": api,
        "debit_modele_par_lot": debit_lots, "charge_progressive": charge,
        "taille_modele_ko": round((root / "models" / "model.pkl").stat().st_size / 1e3, 1),
        "memoire_serveur_mo": memoire_mo, "verdicts": verdicts}, indent=2, ensure_ascii=False))

    df = pd.DataFrame(charge)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["utilisateurs"], df["debit_rps"], "o-", color="#4C72B0", label="Débit (req/s)")
    ax1.set_xlabel("Utilisateurs simultanés")
    ax1.set_ylabel("Débit (requêtes/s)", color="#4C72B0")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(df["utilisateurs"], df["utilisateurs"])
    ax2 = ax1.twinx()
    ax2.plot(df["utilisateurs"], df["p95_ms"], "s--", color="#C44E52", label="P95 (ms)")
    ax2.axhline(OBJECTIFS["charge_p95_ms"], color="#C44E52", lw=0.8, ls=":")
    ax2.set_ylabel("Latence P95 (ms)", color="#C44E52")
    ax1.set_title("Charge progressive : débit et latence P95")
    fig.tight_layout()
    fig.savefig(root / "reports" / "performance_charge.png", dpi=120)
    plt.close(fig)
    assert all(verdicts.values()), verdicts
