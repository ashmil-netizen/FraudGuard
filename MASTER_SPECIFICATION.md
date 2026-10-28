# MASTER PROJECT PROMPT — BUILD A COMPLETE GRAPH-BASED ML FRAUD DETECTION WEB APPLICATION

Act as a senior machine learning engineer, Python backend developer, React frontend developer, data scientist, UI/UX designer, and software architect.

I am a B.Tech Computer Science student specializing in AI and Machine Learning. I need you to help me build a complete, functional, professional-looking academic project from scratch.

Do not create just a prototype, static dashboard, or collection of disconnected code snippets. Build a working end-to-end application with a real dataset, real data processing, a working ML pipeline, graph analysis, a Python API, and a polished frontend.

Explain important technical decisions in beginner-friendly language, but prioritize implementing a working project.

# 1. PROJECT DETAILS

Project title:

"Machine Learning-Based Fraud Detection Using Graph Traversal for Financial Transaction Analysis"

Application name: FraudGuard AI

Project objective:

Build a web application that analyzes e-commerce payment and order-related data, identifies unusual transaction patterns using machine learning, represents relationships between customers, orders, sellers, and payments as a graph, and helps users investigate suspicious patterns using graph traversal algorithms.

The application should demonstrate:

- Data engineering and preprocessing.
- Machine learning-based anomaly detection.
- Graph construction and network analysis.
- Breadth-First Search (BFS).
- Depth-First Search (DFS).
- Interactive graph visualization.
- Transaction-level investigation.
- Model evaluation and explainability.
- A modern, animated, responsive web interface.

IMPORTANT SCIENTIFIC CONSTRAINT:

The Olist Brazilian E-Commerce dataset is not a labeled bank money-laundering dataset. Do not falsely claim that its records are confirmed fraudulent financial transactions.

Inspect the available CSV files and their actual columns before designing the data pipeline.

Use the dataset to demonstrate suspicious-pattern or anomaly detection in e-commerce payment and order relationships.

If genuine fraud labels are unavailable, use an appropriate unsupervised anomaly-detection method, such as Isolation Forest, and clearly describe its output as an anomaly score or a candidate for investigation.

Do not invent fraud labels, claim confirmed fraud, fabricate model accuracy, or describe synthetic labels as real ground truth.

The interface may use the terms "Suspicious Pattern," "Anomaly," and "Flagged for Review." Clearly distinguish these from confirmed fraud.

In the documentation, explain how the project could be extended to a properly labeled financial transaction dataset in the future.

# 2. DATASET DOWNLOAD AND DISCOVERY

Use this KaggleHub dataset:

Dataset identifier: olistbr/brazilian-ecommerce

The download code is:

```python
import kagglehub

# Download the latest available dataset version
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

print("Path to dataset files:", path)
```

Requirements:

1. Create a dataset download script.
2. Download the dataset through KaggleHub.
3. Discover the actual downloaded CSV files automatically.
4. Print the dataset directory and the available filenames.
5. Load the relevant CSV files using Pandas.
6. Inspect the column names, data types, row counts, missing values, duplicate records, and key relationships.
7. Identify which fields are suitable for payment analysis and graph construction.
8. Handle missing values and invalid records appropriately.
9. Merge files only when the join keys and relationship semantics justify it.
10. Prevent accidental many-to-many joins and duplicated payment totals.
11. Convert timestamps and numeric fields appropriately.
12. Create derived features using available data only.
13. Save processed data and a dataset summary for the application.

Potential files may include orders, customers, order payments, order items, sellers, products, and geolocation data. Verify their actual availability rather than assuming every file exists.

Important: Olist's payment records can include multiple payment entries per order. Handle these carefully.

Do not use order review comments, post-order outcomes, or other information unavailable at prediction time as leakage-prone features when claiming to predict risk at an earlier time.

Do not use customer IDs or order IDs as meaningful numerical risk features simply because they are available.

The application must remain functional when some optional CSV files are absent. Display a clear error if essential files are missing.

# 3. REQUIRED TECHNOLOGY STACK

Frontend:
- React
- Vite
- TypeScript
- Tailwind CSS
- React Router
- Framer Motion for animations
- Lucide React for icons
- Recharts for charts
- React Force Graph 2D or another suitable interactive graph visualization library

Backend:
- Python
- FastAPI
- Pandas
- NumPy
- scikit-learn
- NetworkX
- Joblib
- Pydantic

Testing:
- Pytest for backend tests
- Appropriate frontend tests for important UI behavior

Use compatible library versions and provide all required package files.

Prefer local development with a straightforward setup.

Do not require PostgreSQL for the first version. Use the downloaded CSV files, saved model artifacts, and appropriate local persistence where necessary. Structure the application so a database can be added later.

# 4. APPLICATION DESIGN AND VISUAL STYLE

Create a premium cybersecurity and financial intelligence dashboard.

Design direction:
- Dark navy and charcoal backgrounds.
- Electric violet and cyan accents.
- Subtle gradients and glows.
- Crisp typography.
- Modern cards with restrained borders.
- Professional spacing and alignment.
- Clear data visualizations.
- Smooth but lightweight animations.
- Excellent desktop and mobile responsiveness.

Suggested colors:
- Background: #080B14
- Sidebar: #0D1220
- Card: #111827
- Primary accent: #8B5CF6
- Secondary accent: #22D3EE
- Success: #34D399
- Warning: #FBBF24
- High-priority alert: #F87171
- Primary text: #F8FAFC
- Secondary text: #94A3B8

Avoid an overdesigned gaming aesthetic. The result should resemble a professional fraud analytics product.

Use Framer Motion for:
- Page transitions.
- Dashboard card entrance animations.
- Animated number counters.
- Smooth modal transitions.
- Sidebar interactions.
- Loading skeletons.
- Subtle hover effects.
- Graph panel transitions.

Animations must be smooth, restrained, and performant. Respect reduced-motion preferences.

Create a consistent application shell with:
- Collapsible sidebar.
- Top navigation bar.
- Search field where useful.
- Notifications or status indicators where meaningful.
- Dataset/model status.
- Responsive mobile navigation.
- Light/dark mode if practical.

Do not add fake functionality. Every visible button should perform a meaningful action or be explicitly disabled with a helpful explanation.

# 5. REQUIRED FRONTEND PAGES

## PAGE A — OVERVIEW DASHBOARD

Create a polished landing dashboard with real statistics calculated from the loaded dataset and analysis results.

Include KPI cards such as:
- Total orders analyzed.
- Total payment records analyzed.
- Unique customers.
- Unique sellers.
- Number of anomaly candidates.
- Percentage of records flagged for review.

Only show metrics supported by the actual dataset. If a metric cannot be calculated, display "Unavailable" with an explanation.

Add charts:
- Orders or payment activity over time.
- Distribution of payment types.
- Payment value distribution.
- Anomaly score distribution.
- Number of flagged records by selected time period.
- Most connected accounts or entities, if meaningful.

Include a recent investigations table with:
- Record or order ID.
- Relevant customer or seller identifier, appropriately masked if needed.
- Payment value.
- Anomaly score.
- Reason for review.
- Investigation status.
- Button to open details.

Include date filters, search, pagination, and clear empty states.

All metrics and charts must be connected to actual backend responses. Do not hardcode dashboard numbers.

## PAGE B — TRANSACTION / PAYMENT ANALYZER

Create an interactive page where the user can inspect a record or enter details for an illustrative transaction.

Provide two modes:

MODE 1: Analyze an existing dataset record.

- Search by an available order ID or payment record identifier.
- Retrieve the record from the backend.
- Display the relevant payment and order attributes.
- Calculate or retrieve its anomaly score.
- Show its associated customer, seller, and graph relationships where available.

MODE 2: Analyze a manually entered hypothetical transaction.

- Provide fields appropriate to the trained model's actual feature schema.
- Validate the form.
- Clearly identify the input as hypothetical.
- Apply the exact preprocessing and feature transformation used during training.
- Return a meaningful score only if the model supports inference for that input.
- Explain limitations when the input lacks sufficient context.

Do not train on the manually entered record.

Do not pretend that a hypothetical transaction has real graph relationships when it is not linked to the dataset.

If a manually entered transaction cannot be analyzed reliably with the current model, explain why and offer the existing-record analyzer instead.

Results should include:
- Anomaly score or model output.
- A clearly defined risk category based on documented thresholds.
- Contributing features or relevant patterns.
- Related entities where available.
- A recommendation for human review.

Never present an anomaly score as a calibrated probability of fraud unless this is genuinely supported by an appropriate validation procedure.

Use wording such as:
"High anomaly score — investigate this record."

Do not state:
"This transaction is definitely fraudulent."

## PAGE C — INTERACTIVE GRAPH EXPLORER

This is one of the most important parts of the project.

Build an attractive, interactive network visualization using React Force Graph 2D or a suitable equivalent.

Graph requirements:
- Customer nodes.
- Order nodes.
- Seller nodes.
- Payment-related nodes or relationships where appropriate and supported by the data.
- Directed edges where the relationship has a meaningful direction.
- Different node shapes or colors for different entity types.
- Different edge styles for different relationship types.
- A clear legend.
- Zoom, pan, drag, hover tooltips, and click-to-inspect.
- Node and edge details in a side panel.
- Search for entities.
- Filter by entity type.
- Filter or highlight anomaly-related records.
- Show nearby connections up to a selected number of graph hops.
- Reset and fit-to-screen controls.
- A loading state for large graphs.
- Limit or sample the initial graph to avoid rendering every node from a large dataset at once.

Make the graph visually attractive:
- Dark graph canvas.
- Subtle background grid or restrained particle effects if performant.
- Smooth highlighting of selected paths.
- Animated edge flow only if it does not impair readability.
- Carefully chosen node colors.
- Clear labels at suitable zoom levels.
- Smooth transitions when filters change.

Do not color an entire customer or seller as fraudulent merely because one related record has a high anomaly score. Explain what the colors mean.

Provide an information panel showing:
- Selected entity ID.
- Entity type.
- Number of incoming and outgoing relationships where applicable.
- Number of linked orders or payments.
- Related anomaly candidates.
- Connected entities.
- Actions to run BFS or DFS from the selected entity.

## PAGE D — BFS AND DFS INVESTIGATION

Create a dedicated graph investigation interface.

Allow users to:
1. Select a starting customer, seller, or other supported graph node.
2. Select BFS or DFS.
3. Select a maximum traversal depth.
4. Run the traversal.
5. Display the traversal result in order.
6. Highlight the explored nodes and edges on the graph.
7. Show the number of discovered nodes and relationships.
8. Select any result to inspect its details.

Explain the algorithms in simple language within the UI.

BFS:
Explore connected nodes level by level.

DFS:
Follow a path deeper before backtracking.

Clearly distinguish graph reachability from fraud detection. BFS and DFS are investigation tools, not classifiers.

Handle disconnected graphs, invalid starting nodes, cycles, and depth limits safely.

## PAGE E — ANOMALY INVESTIGATIONS

Create a searchable, filterable investigation queue.

Include:
- Record identifier.
- Payment value where applicable.
- Anomaly score.
- Main contributing features.
- Linked entities.
- Detection method.
- Investigation status.
- Timestamp where available.

Add filters for:
- Score or severity band.
- Date range.
- Payment type.
- Investigation status.
- Entity type where applicable.

Allow a user to open a case, inspect graph connections, and change its status between:
- Pending Review.
- Investigating.
- Resolved.
- Dismissed.

Persist these review-status changes locally or through the backend so they survive page refreshes. Do not confuse review status with a ground-truth fraud label.

Provide CSV export of filtered results.

## PAGE F — MODEL PERFORMANCE AND DATASET

Create a page that explains the actual dataset and detection methodology.

Include:
- Dataset source and downloaded files.
- Dataset dimensions.
- Missing-value summary.
- Selected features.
- Preprocessing steps.
- Model type.
- Training or fitting procedure.
- Feature importance or anomaly explanations where supported.
- Model limitations.
- Data leakage precautions.

For unsupervised anomaly detection, do not fabricate accuracy, precision, recall, F1-score, or a confusion matrix when ground-truth labels are unavailable.

Instead, display suitable diagnostics:
- Anomaly score distribution.
- Number and percentage flagged at the selected threshold.
- Sensitivity of flagged counts to threshold changes.
- Manual investigation findings, if genuinely recorded.
- Appropriate validation results when a defensible labeled evaluation dataset exists.

Clearly label unsupervised results as anomaly detection rather than supervised fraud classification.

# 6. MACHINE LEARNING PIPELINE

Create a reproducible backend pipeline.

Suggested stages:

1. Download and discover data.
2. Load relevant CSVs.
3. Validate schema and relationships.
4. Clean and preprocess the data.
5. Create meaningful payment and order-level features.
6. Construct the model feature matrix.
7. Fit an appropriate anomaly-detection model.
8. Calculate anomaly scores.
9. Choose a documented threshold.
10. Store the fitted model and preprocessing configuration.
11. Expose inference through FastAPI.
12. Generate explanations for the records selected for review.
13. Connect the results to the graph analysis.

Consider Isolation Forest as a baseline model because genuine fraud labels may be unavailable.

Choose features based on actual available data, potentially including:
- Payment value.
- Number of payment entries per order.
- Payment type.
- Number of installments.
- Order item count.
- Number of sellers associated with an order.
- Customer purchase frequency, if constructed without leakage.
- Time-related features where valid.
- Other meaningful aggregated features supported by the dataset.

Do not assume all these columns exist in one CSV. Create them through valid joins or aggregations when justified.

Avoid using raw high-cardinality identifiers as numeric model features.

Avoid target leakage and train/test contamination where the methodology requires a held-out evaluation.

Document why each feature is included.

For Isolation Forest, use `decision_function` or `score_samples` consistently and document the score direction. Convert the raw score into a clear UI representation if helpful, without falsely interpreting it as a fraud probability.

Allow the user to adjust the anomaly review threshold through the UI if practical.

When the threshold changes, update the flagged-record count and graph highlighting consistently.

Save the model with Joblib. Include a reliable way to check whether the saved model and feature schema match the current dataset.

Do not fit a fresh model on every frontend request.

# 7. GRAPH CONSTRUCTION

Use NetworkX in the Python backend to construct a graph from valid dataset relationships.

Model the real entities and relationships present in the data. For example:
- Customer → Order.
- Order → Seller through order-item relationships.
- Order → Payment record.

Only use relationships supported by the dataset. Explain that these are e-commerce relationships rather than direct bank-account transfers.

Choose a suitable graph representation and preserve important relationship attributes.

If the same two entities have multiple distinct relationships, do not silently overwrite them.

Store relevant attributes such as:
- Entity type.
- Relationship type.
- Order ID.
- Payment value when applicable.
- Anomaly score when associated with a record.
- Timestamp when available.

Implement efficient BFS and DFS endpoints with:
- Starting node.
- Traversal algorithm.
- Maximum depth.
- Resulting node list.
- Resulting edge list.
- Traversal statistics.

Provide an endpoint to retrieve a graph neighborhood around a selected entity.

Avoid returning the entire graph for every request. Support pagination, bounded neighborhoods, or server-side graph filtering.

Keep graph traversal logic separate from anomaly scoring logic.

# 8. FASTAPI BACKEND

Build a well-structured API with validation and clear error messages.

Suggested endpoints:

GET /api/health
Return application and model status.

GET /api/dataset/summary
Return dataset statistics and available files.

GET /api/dashboard/metrics
Return real dashboard metrics.

GET /api/dashboard/charts
Return data for dashboard charts.

GET /api/records
Return searchable, filterable, paginated records.

GET /api/records/{record_id}
Return details for one existing record.

POST /api/analyze
Analyze a valid existing or supported hypothetical record.

GET /api/anomalies
Return anomaly candidates using the selected threshold.

GET /api/graph
Return a bounded graph or a filtered graph view.

GET /api/graph/node/{node_id}
Return node details and nearby relationships.

POST /api/graph/traverse
Run BFS or DFS with a start node and depth limit.

GET /api/model/summary
Return model metadata, feature schema, threshold, and supported diagnostics.

GET /api/export/anomalies
Export the filtered anomaly results as CSV.

POST /api/investigations/{record_id}/status
Update the investigation status.

You may adjust endpoint designs to fit the data and implementation. Do not create endpoints that return fabricated data.

Add:
- Pydantic request and response schemas.
- CORS configuration restricted appropriately for local development.
- Input validation.
- Exception handling.
- Logging.
- Pagination.
- Reasonable request limits.
- API documentation through FastAPI's built-in OpenAPI interface.

If an endpoint cannot be implemented because the dataset lacks necessary information, explain the limitation and provide an appropriate alternative.

# 9. FRONTEND AND BACKEND INTEGRATION

The React application must consume the real FastAPI endpoints.

Requirements:
- Centralized API client.
- Typed request and response objects.
- Loading, success, empty, and error states.
- Reusable components.
- Search and filters connected to actual backend queries.
- Charts driven by returned data.
- Graph nodes and edges from the backend.
- BFS/DFS results from the backend.
- Real model results from the backend.
- Persistent investigation status.
- Working CSV export.

Do not hardcode example metrics into the production dashboard.

If demonstration data is needed during development, isolate it in an explicitly labeled demo mode. The application must clearly indicate when it is showing demo data rather than dataset-derived results.

# 10. PROJECT STRUCTURE

Organize the project cleanly. For example:

```text
fraudguard-ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── services/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── graph/
│   │   ├── ml/
│   │   └── utils/
│   ├── scripts/
│   │   ├── download_dataset.py
│   │   ├── inspect_dataset.py
│   │   ├── preprocess.py
│   │   ├── train_model.py
│   │   └── build_graph.py
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── layouts/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   └── App.tsx
│   ├── package.json
│   └── ...
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── README.md
└── .gitignore
```

You can improve this structure if needed.

Never commit large datasets, credentials, virtual environments, node_modules, or private local configuration files to Git.

Do not place Kaggle credentials directly in source code.

# 11. ERROR HANDLING AND SECURITY

The application should handle:
- Kaggle download failures.
- Missing CSV files.
- Unexpected column names.
- Empty datasets.
- Missing values.
- Invalid user inputs.
- Unknown record IDs.
- Unknown graph node IDs.
- Model artifacts missing or incompatible with the dataset.
- Backend connection failures.
- Oversized graph requests.

Display readable error messages in the frontend.

Keep secrets out of source code. Validate all incoming requests. Do not expose local filesystem paths or environment secrets through the API.

Mask customer-identifying details where appropriate and avoid unnecessary exposure of personal information.

# 12. TESTING

Create meaningful tests for:
- Dataset loading and schema validation.
- Feature generation.
- Model fitting and scoring.
- Threshold behavior.
- Graph construction.
- Repeated relationships.
- BFS and DFS.
- API response schemas.
- Invalid request handling.
- Investigation status persistence.
- Frontend loading and error states.

Tests must not depend on Kaggle being available every time they run. Use small, clearly identified fixtures for unit tests.

Include a smoke test that verifies the main workflow works from dataset preparation to API response.

# 13. DOCUMENTATION AND PRESENTATION

Create a professional README explaining:
- Project objective.
- Architecture.
- Dataset source.
- Dataset limitations.
- Technology stack.
- Installation instructions.
- Dataset download instructions.
- Model training.
- Backend startup.
- Frontend startup.
- API documentation.
- Testing.
- How to use the UI.
- Model limitations.
- Potential future improvements.

Include a Mermaid architecture diagram if useful.

Also create a concise academic methodology document describing:
- Problem statement.
- Existing limitations.
- Proposed system.
- Dataset and preprocessing.
- Model methodology.
- Graph representation.
- BFS and DFS.
- Evaluation strategy.
- Limitations and future scope.

Do not claim novelty merely because standard algorithms are combined. Identify the actual contribution honestly, such as an integrated, explainable e-commerce anomaly investigation workflow, if the implementation supports it.

# 14. IMPLEMENTATION WORKFLOW — FOLLOW THIS ORDER

Do not attempt to dump the entire project in one giant answer without validating it.

PHASE 1 — Inspect and plan:
- Explain the architecture.
- Create the initial folder structure.
- Create the dataset downloader.
- Download or inspect the actual dataset if your execution environment permits.
- List the real CSV files and columns.
- Decide the feature and graph schemas based on the findings.

PHASE 2 — Data pipeline:
- Implement preprocessing.
- Validate joins.
- Create model features.
- Build the graph.
- Train and save the anomaly-detection model.
- Verify the outputs.

PHASE 3 — Backend:
- Implement FastAPI endpoints.
- Add validation and error handling.
- Run backend tests.
- Confirm API responses using actual processed data.

PHASE 4 — Frontend:
- Build the application shell.
- Implement the dashboard.
- Implement the analyzer.
- Implement the interactive graph explorer.
- Implement BFS/DFS investigation.
- Implement the investigation queue.
- Implement the dataset and model page.
- Connect every page to the API.

PHASE 5 — Integration:
- Run frontend and backend together.
- Test the main user workflow.
- Fix integration and runtime errors.
- Verify that dashboard numbers, anomaly results, and graph data agree.
- Verify all visible controls.

PHASE 6 — Final polish:
- Improve animations and responsive layouts.
- Improve graph readability and performance.
- Add documentation.
- Provide a complete local setup guide.
- Summarize known limitations honestly.

# 15. HOW I WANT YOU TO WORK

I will use Claude Code or an AI coding assistant in VS Code to create and run the project.

If you can directly access the project workspace and execute commands, create the actual files and test them. Otherwise, give me the exact files and commands needed, in manageable stages.

Do not claim you downloaded a dataset, ran a test, or launched the application unless you actually did so.

Do not stop after generating the frontend. Continue through backend implementation and integration.

Do not leave major features as TODOs or static mockups.

If a library API or dependency is uncertain, verify it or choose a reliable alternative.

If you discover that an intended feature cannot be supported by the Olist data, explain the limitation and implement the closest scientifically valid alternative.

Start with PHASE 1. First inspect the environment, create the project structure, and download and inspect the actual dataset. Then proceed step by step, verifying each phase before moving to the next.

At the end, I should have a complete, runnable, visually impressive graph-based e-commerce anomaly detection and investigation platform, with a working ML pipeline, BFS/DFS, interactive network visualization, API, tests, and documentation.