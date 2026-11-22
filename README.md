# CreditRisk-Test

Stratégie de test d'un système d'estimation du risque de défaut de remboursement (microfinance) :
validation des demandes, prétraitement, régression logistique et API de prédiction, testés à chaque
modification du dépôt par GitHub Actions.

Les données sont **synthétiques et volontairement biaisées** (variante du German Credit publiée par
IBM, voir la section « Données ») : elles servent à tester le système et ne démontrent pas la performance
réelle d'un modèle de crédit.

## Structure

```
CreditRisk-Test/
├── app/
│   ├── main.py            API FastAPI : GET /health, POST /predict, GET /model-info
│   ├── validation.py      contrat d'entrée : rejet des demandes invalides, alertes
│   ├── preprocessing.py   chargement, préparation, découpage 60/20/20
│   └── prediction.py      modèle : entraînement, seuil choisi sur la validation, métriques,
│                          non-régression, sauvegarde et chargement, prédiction
├── tests/
│   ├── test_validation.py définit aussi les données et fixtures de test partagées
│   ├── test_api.py
│   ├── test_integration.py
│   ├── test_robustness.py
│   └── test_regression.py
├── data/raw/german_credit_data.csv   jeu de données (911 Ko, voir « Données »)
├── models/model.pkl                  modèle de référence versionné
├── reports/               métriques, référence, mesures (performance, figures)
├── docs/                  plan de test, rapports (Markdown et Word), non versionné
├── .github/workflows/tests.yml
├── requirements.txt       versions figées
├── pyproject.toml         configuration de Pytest et de Ruff
└── README.md
```

## Données

- **Source** : variante synthétique du German Credit publiée par IBM (démonstrations Watson OpenScale),
  complétée par une colonne `MonthlyIncome` calculée (voir ci-dessous).
  Les données sont **synthétiques et volontairement biaisées** : elles servent à tester le système
  et ne permettent pas de démontrer la performance réelle d'un modèle de crédit.
- **Emplacement** : `data/raw/german_credit_data.csv` (911 Ko). Facultatif pour la CI : s'il est
  absent du dépôt, les 2 tests qui l'utilisent sont ignorés, les autres utilisent des données synthétiques.
- **Empreinte SHA-256 du fichier de référence** :
  `ce1e9a11706c22f9339688cd23a42eae97e058909447de46a5ca40ef827a5445`
  (vérification : `sha256sum data/raw/german_credit_data.csv`)
- **Taille** : 5 000 lignes, 23 colonnes, aucune valeur manquante, 18 doublons hors identifiant.
- **Encodage** : UTF-8 avec BOM en début de fichier (lu correctement par pandas).

### Colonnes

| Rôle | Colonnes |
|---|---|
| Identifiant (exclu du modèle) | `CustomerID` |
| Cible | `Risk` : « Risk » → `defaut = 1` (1 670 lignes, 33,4 %), « No Risk » → `defaut = 0` (3 330 lignes) |
| Numériques (8) | `LoanDuration`, `LoanAmount`, `InstallmentPercent`, `CurrentResidenceDuration`, `Age`, `ExistingCreditsCount`, `Dependents`, `MonthlyIncome` |
| Catégorielles (13) | `CheckingStatus`, `CreditHistory`, `LoanPurpose`, `ExistingSavings`, `EmploymentDuration`, `Sex`, `OthersOnLoan`, `OwnsProperty`, `InstallmentPlans`, `Housing`, `Job`, `Telephone`, `ForeignWorker` |
| Attributs sensibles | `Age`, `Sex`, `ForeignWorker` |

Les 21 variables explicatives sont toutes utilisées par le modèle. Les modalités autorisées sont
définies dans `app/validation.py` (`CATEGORIES`).

### Calcul du revenu mensuel

Avec un taux annuel de 10 %, soit r = 0,10 / 12, et n = `LoanDuration` (en mois) :

- mensualité = `LoanAmount` × r(1 + r)^n / ((1 + r)^n − 1) (amortissement à mensualités constantes) ;
- `MonthlyIncome` = mensualité / part du revenu consacrée à la mensualité, arrondi à deux décimales.

Hypothèse : `InstallmentPercent` est un code de tranche, comme dans le German Credit d'origine, où la
part du revenu diminue quand le code augmente. On retient le centre de chaque tranche, et on prolonge
l'échelle pour les codes 5 et 6, absents de l'original :

| `InstallmentPercent` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Tranche d'origine | 35 % et plus | 25 à 35 % | 20 à 25 % | moins de 20 % | – | – |
| Part retenue | 35 % | 30 % | 22,5 % | 17,5 % | 12,5 % | 7,5 % |

La fonction `expected_income` de `app/validation.py` reproduit ce calcul (écart maximal 0,005, dû à
l'arrondi). La version précédente de la colonne (`revenu mensuel`, hypothèse « 3 = 3 % du revenu »)
a été remplacée.

Valeurs obtenues : de 23,61 à 6 594,01, médiane 786,46 (même unité que `LoanAmount`, pas en FCFA).
Revenu moyen : 1 207 pour les dossiers en défaut, 770 pour les autres.

**Limites** :
- le revenu est **dérivé** de `LoanAmount`, `LoanDuration` et `InstallmentPercent` ; il n'apporte pas
  d'information observée nouvelle au modèle ;
- la part retenue est le centre d'une tranche : le revenu réel d'un demandeur peut s'en écarter, et les
  parts des codes 5 et 6 sont une extrapolation ;
- les dossiers en défaut ont un revenu moyen plus élevé, contrairement à l'intuition métier : ils ont
  des montants plus élevés et des codes `InstallmentPercent` plus grands (part du revenu plus faible).

### Correspondance avec les variables du sujet

| Sujet | Jeu de données |
|---|---|
| `age` | `Age` |
| `revenu_mensuel` | `MonthlyIncome` (calculé) |
| `montant_credit` | `LoanAmount` (unité non précisée, pas en FCFA) |
| `duree_mois` | `LoanDuration` |
| `anciennete_activite` | `EmploymentDuration` (5 tranches, pas en années) |
| `defaut` | `Risk` converti en 0/1 |

## Exécution locale

Prérequis : Python 3.12.

```bash
# Environnement reproductible (versions figées)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Qualité du code
ruff check .

# Tests et rapport de couverture (terminal + reports/coverage/index.html + reports/junit.xml)
pytest

# Un seul fichier de tests
pytest tests/test_api.py

# Mesures longues sur le jeu réel, exclues par défaut et de la CI (quelques minutes) : exploration,
# versions candidates, performance complète ; régénèrent les fichiers de reports/
# (la performance démarre un serveur uvicorn à 2 processus sur le port 8001, qui doit être libre)
pytest -m mesures --no-cov

# Réentraîner le modèle et régénérer les rapports (models/model.pkl, reports/metrics.json…)
python -m app.prediction
# Créer la référence de non-régression (refusé si reports/baseline.json existe déjà)
python -m app.prediction --save-baseline

# Lancer l'API (2 processus) puis l'interroger (documentation interactive : http://127.0.0.1:8001/docs)
uvicorn app.main:app --port 8001 --workers 2
curl http://127.0.0.1:8001/health
```

## Intégration continue

Le workflow `.github/workflows/tests.yml` s'exécute à chaque Pull Request, à chaque envoi sur
`main`, et à la demande (`workflow_dispatch`). Il :

1. installe Python 3.12 et les dépendances figées (avec cache pip) ;
2. contrôle la qualité du code avec Ruff ;
3. exécute les tests avec Pytest et mesure la couverture avec pytest-cov (sans le jeu de données
   réel, les 2 tests qui l'utilisent sont ignorés) ;
4. affiche un résumé (tests réussis, couverture) dans l'onglet du workflow ;
5. publie les rapports (`junit.xml`, `coverage.xml`, rapport HTML) en artefact.

Durée : moins d'une minute de tests. Les tests marqués `mesures` ne sont pas exécutés. Aucun
réentraînement complet : les tests utilisent des modèles synthétiques entraînés en quelques
millisecondes, et le modèle versionné `models/model.pkl`, comparé à `reports/baseline.json`.

### Vérifier qu'un test défaillant fait échouer le workflow

```bash
git switch -c test/echec-volontaire
# Désactiver un contrôle réel : l'âge minimal n'est plus vérifié par l'API
sed -i 's/Field(ge=AGE_RANGE\[0\], le=AGE_RANGE\[1\], /Field(le=AGE_RANGE[1], /' app/main.py
git commit -am "Test : régression volontaire (âge minimal non contrôlé)"
git push -u origin test/echec-volontaire
# Ouvrir la Pull Request : le workflow doit échouer (test_age_negatif et d'autres)
# Puis fermer la Pull Request SANS la fusionner et supprimer la branche
```

## Documentation

Le dossier `docs/` est exclu du dépôt (`.gitignore`) : les documents ci-dessous sont remis
séparément.

| Document | Contenu |
|---|---|
| `docs/plan_de_test` | Exigences EX01 à EX17, cas de test, matrice de traçabilité, critères d'acceptation |
| `docs/resultats_tests_api` | Contrat de l'API, scénarios S1 à S10 et résultats |
| `docs/resultats_tests_integration` | Scénario de bout en bout et gestion des erreurs |
| `docs/registre_anomalies` | Tests de robustesse et registre des anomalies A1 à A14 |
| `docs/rapport_non_regression` | Référence, tolérances, comparaison de versions |
| `docs/rapport_performance` | Latence, débit, charge progressive |
| `docs/rapport_final_qualification` | Rapport de qualification : synthèse, traçabilité, anomalies, décision |
| `docs/architecture.png` | Schéma d'architecture (source draw.io : `docs/architecture.drawio`) |
