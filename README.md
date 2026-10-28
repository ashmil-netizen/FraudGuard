# FraudGuard AI

**Machine Learning-Based Fraud Detection Using Graph Traversal for Financial Transaction Analysis**

FraudGuard AI is an academic demonstration that combines a supervised
transaction classifier with account-network exploration. It uses PaySim
simulation data, a scikit-learn Random Forest, a FastAPI backend, and a React
dashboard. Investigators can search transactions, inspect bounded graph
neighborhoods, and explore account connections with breadth-first search (BFS)
or depth-first search (DFS).

> **Important:** PaySim is simulated mobile-money data, not real bank-customer
> activity. `isFraud` is a simulation label, not independently confirmed
> real-world fraud. Model scores are uncalibrated positive-class scores, not
> guaranteed or calibrated fraud probabilities. The model includes
> post-transaction balances, so its prediction point is after transaction
> posting. Graph connections and traversal results are investigation clues,
> not fraud determinations.

## What is in this repository?

- `backend/scripts/` — download, inspect, preprocess, train, evaluate, and graph
  investigation scripts.
- `backend/app/ml_pipeline.py` — validated preprocessing and model pipeline.
- `backend/app/transaction_graph.py` — directed multi-graph construction and
  reusable BFS/DFS traversals.
- `backend/app/api/` — FastAPI routes, request/response schemas, and services.
- `backend/tests/` — tests for model/data handling, graph traversal, and API.
- `frontend/src/` — typed API client and six dashboard pages.
- `MASTER_SPECIFICATION.md` — original project brief; some early requirements
  describe a different dataset. The PaySim implementation and verified facts
  in this README describe the current project.

## Run from a fresh clone

### 1. Clone the repository

```shell
git clone https://github.com/ashmil-netizen/FraudGuard.git
cd FraudGuard
```

### 2. Create the Python environment

Use Python 3.11 or later and Node.js 20.19+ (or 22.12+).

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal,
then activate again. Alternatively, call `.venv\Scripts\python.exe` directly.

**macOS / Linux:**

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### 3. Download PaySim and prepare the model artifacts

The dataset and generated model/data files are intentionally not stored in
GitHub. Download the dataset with KaggleHub, then run the pipeline from the
repository root:

```shell
python backend/scripts/download_dataset.py
python backend/scripts/inspect_dataset.py
python backend/scripts/preprocess.py
python backend/scripts/train_model.py
python backend/scripts/evaluate_model.py
```

The first download requires internet access and may require Kaggle account
access. If KaggleHub requests authentication, authenticate locally with
`python -c "import kagglehub; kagglehub.login()"` and follow its prompts. Do not
put Kaggle credentials in this repository. The download script discovers the
dataset directory returned by KaggleHub and reuses its local cache. The
preprocessor streams the source file in chunks and leaves the original
download unchanged.

The data download is about 500 MB; preprocessing creates additional large
files and the API builds a local SQLite index on first startup. Allow several
GB of free disk space. Training the Random Forest can take several minutes
and requires additional memory.

### 4. Start the API

Open a terminal at the repository root, activate the Python environment, and
run:

```shell
python -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000
```

The API documentation is at <http://127.0.0.1:8000/docs>. The service loads
the saved model and uses the scored chronological test split; it does not
retrain on API requests. On first startup, it creates a local SQLite index
from graph test transactions.

### 5. Start the frontend

In another terminal:

```shell
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. The default backend URL is
`http://127.0.0.1:8000`. To use another API origin, copy
`frontend/.env.example` to `frontend/.env.local`, set
`VITE_API_BASE_URL`, and restart Vite. CORS is restricted to the local
development origins documented below.

### 6. Run tests and build

From the repository root, with the Python environment active:

```shell
python -m pytest backend/tests -q
```

From `frontend/`:

```shell
npm test
npm run build
```

The frontend test/build commands do not require the PaySim download. Backend
tests use small fixtures where possible; running the full application does
require the dataset and generated artifacts.

## Active outputs

- `backend\reports\dataset_inspection.md` and `.json` — actual source schema,
  dimensions, quality checks, label distribution, and numeric summaries.
- `backend\reports\dataset_analysis.md` — verified relationships, leakage
  cautions, and the next-phase graph design.
- `backend\data\processed\processed_transactions.csv` — transaction lookup
  fields and split membership.
- `backend\data\processed\feature_matrix.csv` and `labels.csv` — model inputs
  and the target stored separately.
- `backend\data\processed\graph_transactions.csv` — one offline graph-preparation
  row per transaction, retaining account IDs, time step, label, and split.
- `backend\data\processed\feature_schema.json` — ordered feature schema,
  definitions, preprocessing rules, and split facts.
- `backend\models\fraud_classifier.joblib` and `model_metadata.json` — the
  reusable fitted pipeline and dataset/schema/threshold metadata.
- `backend\data\processed\scored_test_transactions.csv` and
  `graph_test_transactions.csv` — held-out scores and transaction graph data
  for offline evaluation.
- `backend\reports\preprocessing_report.md`, `model_training_summary.md`, and
  `model_report.md` — preprocessing facts, validation-based threshold choice,
  and final held-out metrics.

The `backend\data\raw\` directory, processed data, models, and reports are
excluded from Git because they are large and can be generated by following the
steps above. The original [master specification](./MASTER_SPECIFICATION.md)
remains as a project reference; its earlier Olist-oriented requirements are
not the active dataset or pipeline.

## Architecture and data flow

```text
KaggleHub PaySim CSV
    -> schema inspection and chunked preprocessing
    -> chronological train / validation / test split
    -> fitted Random Forest + saved feature metadata
    -> held-out test predictions and graph transaction data
    -> FastAPI services and bounded SQLite-backed queries
    -> typed React client, charts, transaction analysis, and graph investigation
```

The API reuses saved artifacts and does not train a model per request.

## Dataset and methodology

The inspected file is
`PS_20174392719_1491204439457_log.csv`: **6,362,620 rows × 11 columns**.
It contains `step`, `type`, `amount`, sender and recipient identifiers,
before/after balances, `isFraud`, and `isFlaggedFraud`. The inspection found
8,213 positive labels, 0 missing values, and 0 duplicate rows by the reported
streaming fingerprint check.

The active model features are:

```text
step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,
newbalanceDest, type
```

Account identifiers are preserved for lookup/graph work but never used as
numeric model features. `isFraud` is the target and is excluded from the feature
matrix. `isFlaggedFraud` is also excluded because it is an existing detection
rule. Invalid core rows are discarded; invalid balance measurements are marked
missing and imputed by a median learned from the training sample. Categorical
transaction types are one-hot encoded in the saved scikit-learn pipeline.

Train, validation, and test periods are split chronologically by complete
`step` values. Row-volume targets are approximately 70/15/15; the actual step
boundaries and counts are recorded in metadata. The 100-tree Random Forest uses
`class_weight="balanced_subsample"`, `max_depth=16`, `min_samples_leaf=2`,
`max_features="sqrt"`, and random seed `42`. A deterministic, class-stratified
sample capped at 500,000 rows is used from the training period for practical
resource use. Preprocessing/encoding is fit only on this training sample.

The classification threshold is selected by maximum F1 on the validation
period only; the final test period is not used to tune it. Accuracy is not
reported because the positive class is rare and accuracy can conceal missed
fraud cases. Test precision, recall, F1, confusion matrix, average precision,
ROC-AUC, and class support are in `backend\reports\model_report.md`.

The graph preparation preserves every transaction as a separate event.
The graph implementation provides a directed NetworkX `MultiDiGraph`: account IDs are nodes
and each transaction is a separate keyed edge directed from sender to recipient.
The transaction ID is the multiedge key, so repeated transfers between a pair
are preserved with their own step, amount, type, and (where present) model score
and predicted class. Offline simulation labels are optional edge attributes and
are omitted by default. An account connected to a flagged transaction is not
thereby proven fraudulent.

## Graph investigation with BFS and DFS

The reusable graph services are in
`backend\app\transaction_graph.py`. Breadth-first search (BFS) explores
outgoing, incoming, or both directions one depth level at a time. Depth-first
search (DFS) follows one path before backtracking. Both track visited accounts
to handle cycles and return account order, depth, connecting transaction
records, and traversal limits/statistics. Neither traversal makes a fraud
decision.

The graph CLI uses only the held-out test period for which saved-model
predictions exist. It scans that CSV in chunks for the requested account
neighborhood, verifies the saved model metadata and threshold, and fails
explicitly if the requested neighborhood exceeds its transaction cap rather
than silently dropping relationships:

```powershell
python backend\scripts\graph_traversal.py --start-account C123 --algorithm bfs --max-depth 2 --direction out --minimum-score 0.5
```

Replace `C123` with an account identifier present in the scored test data. Use
`--algorithm dfs`, `--direction in` or `both`, or change `--max-depth` as
needed. Default output is `backend\reports\graph_investigation.json`. To include
the simulation label for offline evaluation only, explicitly pass
`--include-ground-truth`; do not expose that option in a real inference UI.

The loader rescans the scored CSV once per requested depth and holds only the
bounded local neighborhood in memory. Defaults cap that neighborhood at 50,000
transactions, traversal results at 10,000 accounts/50,000 edges, and use only
transactions from the existing scored test period. A neighborhood outside
that time period is not visible, and the graph is not a full-network view.
Raising caps increases memory use; a larger deployment should use indexed
persistent adjacency storage.

## FastAPI backend

The API reuses the saved model, training feature order, test-split model scores,
and the Phase 3 graph/traversal functions. Start it from the project directory:

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI interface. On first
startup, the API streams `graph_test_transactions.csv` into a local SQLite
index at `backend\data\processed\api_transactions.sqlite3`; subsequent starts
reuse it unless the source file changes. The index contains only scored
chronological test-split transactions, not the full six-million-row dataset,
and can require additional disk space. It is generated locally and is not
committed with the project.

Available routes include `/api/health`, `/api/dataset/summary`,
`/api/dashboard/metrics`, transaction list/detail and prediction routes,
bounded `/api/graph` and account routes, BFS/DFS traversal at
`/api/graph/traverse`, and `/api/model/summary`. The API never retrains the
classifier. Predictions for request-submitted transactions are explicitly
hypothetical; scores are uncalibrated positive-class scores, not calibrated
fraud probabilities. Ground-truth labels are kept out of transaction and graph
responses, and graph traversal indicates connectivity rather than fraud.

Pagination is capped at 200 items; graph results are capped at 5,000
transactions and traversal depth at six hops. CORS is limited to the local
development origins `http://localhost:5173` and `http://127.0.0.1:5173`.
Run the backend tests with:

```powershell
python -m pytest backend\tests -q
```

## React frontend

The React/Vite client is in `frontend\`. Install dependencies and start the
development server from that directory:

```powershell
cd frontend
npm ci
npm run dev
```

By default, the client uses `http://127.0.0.1:8000` as its API origin. Copy
`.env.example` to `.env.local` and set `VITE_API_BASE_URL` when the backend
runs elsewhere. Start the API separately as described above; its CORS allowlist
permits the Vite localhost development origins only.

The frontend includes the overview dashboard, existing/hypothetical transaction
analyzer, bounded graph explorer, BFS/DFS investigation page, predicted
candidate queue, and dataset/model information page. Metrics and records are
loaded from the API. Graph and transaction data are limited to the scored test
split. Investigation status persistence and CSV export are not provided by the
backend, so the corresponding queue controls are explicitly unavailable.

Run frontend checks:

```powershell
cd frontend
npm test
npm run build
```

## Reproduce the current saved-model run

The exact run produced:

- Training rows available / used: **4,463,587 / 499,999**.
- Validation rows: **980,416**.
- Held-out test rows: **918,617**, including **4,006** simulated fraud labels.
- Validation-selected threshold: **0.25991107**.
- Held-out precision **0.99763033**, recall **0.73564653**, F1 **0.84683908**,
  average precision **0.88463254**, and ROC-AUC **0.99088888**.
- Confusion matrix (TN, FP, FN, TP): **914,604, 7, 1,059, 2,947**.

These are measured results for this simulated dataset and this temporal split;
they do not establish effectiveness on real financial activity. The saved
model is reloaded by the evaluator and validated against its PaySim identifier,
model type, and exact ordered feature schema.

colab - https://colab.research.google.com/drive/1fbDJ7tfJOWx4I8BETOuMyNIDDYoFMt8-?usp=sharing
