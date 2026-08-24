# Data Model Explorer (DaMoOp*)

A Streamlit dashboard + CLI toolkit for exploring alternative NoSQL data-model designs and estimating their **time**, **financial**, and **environmental** costs.

It drives a two-phase Java pipeline over a base TPC-C schema:

1. **`DMgeneration.jar`** — applies merge/split transformations to the base model and generates ~135 candidate data models (as XML), plus a signatures file describing each one.
2. **`CostCalculation.jar`** — computes, for every generated model, the cost of executing a 20-query workload across a grid of data volumes (1K–100M warehouses) and server counts (10–10,000), writing a `costs.txt` per model.

On top of that pipeline, the project adds:

- an interactive **dashboard** to run the pipeline, edit inputs, and analyze/rank the results (Pareto frontiers, stability scoring, transformation-tree drill-down);
- a **parametric profile sweep** (Latin Hypercube over 18 schema dimensions) that batch-runs the pipeline over hundreds of generated schema profiles to build ML training datasets;
- **ML models** trained on those datasets: a cost *predictor* (config → cost) and a model *recommender/ranker* (config + workload → best data model out of all 135, ranked), both usable live from the Predict page with one-click validation against the real JAR pipeline.

## Dashboard Pages

| Page | What it does |
|---|---|
| **Input Configuration** | Edit queries, config, schema keys, and indexes; save/activate named profiles (A, B, C, …) |
| **Pipeline** | Run Phase 1 (DMG: generate model variants) and Phase 2 (Cost JAR) with live logs, progress, and inactivity timeout; snapshot runs |
| **Cost Analysis** | Compare all models: best-model cards, stability leaderboard (Eq. 14–17), Pareto trade-offs, per-query breakdowns, transformation tree |
| **Profile Sweep** | Latin-Hypercube-sample N schema profiles (18 dims), batch-run the pipeline per profile, assemble the ML dataset (`Output/sweep_ml.parquet`) |
| **Scenario Analysis** | Generate (volume, servers, query-frequency) scenarios, explore diversity, export flat ML training data, build the temporal scenario dataset |
| **Predict** | *Recommend best model* — rank all 135 models for a given schema config + workload using the trained ranker; *Predict one model* — instant cost prediction. Both with one-click validation against the real pipeline |
| **History** | Manage and compare saved cost snapshots |

## Project Layout

- `app.py` — Streamlit entry point (config, sidebar, routing)
- `lib/` — shared code: constants/paths/CSS, data loading & caching, `costs.txt` parsing, Eq. 14–17 scoring, Pareto/tree helpers, scenario reweighting, UI-state persistence
- `views/` — one module per dashboard page
- `scripts/` — CLI tools (all runnable as `python -m scripts.<name>`):
  - `sweep_profiles` — LHS-sample N parametric profiles (deterministic per seed)
  - `run_sweep` — run DMG + Cost JARs over every profile, archiving each result
  - `assemble_dataset` — parse all archived costs into one Parquet dataset (parallel, streaming)
  - `validate_dataset` / `validate_scenario_dataset` — PASS/FAIL sanity batteries for the datasets
  - `scenario_sweep` — expand the structural dataset across M query-frequency mixes (pure Python reweighting, no JAR runs)
  - `train_predictor` — XGBoost cost predictor (config → total costs)
  - `train_ranker` — XGBoost unit-cost ranker (config → 60 per-query unit costs + overheads; workload applied via a closed form)
  - `verify_closed_form` — proves the closed-form workload reweighting matches the real Cost JAR
  - `loqo_gate` — leave-queries-out experiment: can a query-featurized model rank the 135 models for *unseen* queries?
  - `simulate_one` — ground-truth run for a single config (used by the Predict page's Validate buttons)
- `jars/` — pipeline executables (`DMgeneration.jar`, `CostCalculation.jar`)
- `Input/` — editable configuration (`queries.txt`, `config.txt`, `keys.txt`, `index.txt`, `queries_dmg.txt`), base model `Input/models/TPC_C.xml`, saved profiles under `Input/profiles/`
- `conf/` — **what `CostCalculation.jar` actually reads.** The UI and sweep scripts sync `Input/` → `conf/` automatically before Phase 2; if you drive the JAR by hand, copy the four config files (`config.txt`, `keys.txt`, `index.txt`, `queries.txt`) there yourself
- `Output/` — generated models and `costs.txt` results (`Output/GeneratedDMs/`). The trained recommender + legacy predictor ship here **compressed** (`Output/models/*.pkl.xz`); the 2.8 GB assembled dataset, sweep archives, and run snapshots are git-ignored (regenerable)
- `report/` — the IEEE paper (LaTeX under `main/`, `paper/`) plus the analysis/figure scripts (`*.py`) and their small outputs that reproduce every figure without the full dataset
- `notebooks/` — EDA and modeling experiments (`ML_500.ipynb` is the EDA gate over the 500-profile sweep)
- `scenarios/` — persisted scenario definitions

## Requirements

- Python 3.10+
- Java 8+ on `PATH` (for the pipeline JARs)
- Python packages: see `requirements.txt` (Streamlit/pandas/plotly for the dashboard; pyarrow for datasets; xgboost/scikit-learn/scipy for training and the Predict page)

## Setup

```sh
git clone https://github.com/manpatell/Data-Model-Explorer.git
cd Data-Model-Explorer
python -m venv .venv
.venv\Scripts\activate        # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

The trained recommender and legacy predictor **ship with the repo**, LZMA-compressed under `Output/models/*.pkl.xz` (the raw ranker is 366 MB — too big for GitHub — so it is committed at ~94 MB). The **Predict** page decompresses them on first use, so it works on a fresh clone with **no dataset generation or training**. To pre-extract them for the CLI/paper scripts:

```sh
python -m scripts.setup_models     # Output/models/*.pkl.xz -> *.pkl
```

## Running

**Dashboard:**

```sh
streamlit run app.py
```

Start on the **Pipeline** page: click *Run Full Pipeline* to generate the 135 models and their costs (a few minutes), then explore them on **Cost Analysis**. The **Predict** page works immediately (models are shipped) — no pipeline run required.

**ML pipeline (CLI, recommended for long batch runs):**

```sh
# 1. Generate N parametric profiles (deterministic per seed)
python -m scripts.sweep_profiles --n 500 --seed 42 --out Input/profiles/sweep

# 2. Run DMG + Cost JARs over every profile (~6-10 s each; resumable)
python -m scripts.run_sweep --sweep-dir Input/profiles/sweep --archive-dir Output/sweep --skip-existing

# 3. Assemble + validate the structural dataset
python -m scripts.assemble_dataset --archive-dir Output/sweep --out Output/sweep_ml.parquet
python -m scripts.validate_dataset

# 4a. Recommender path: verify the closed form, then train the ranker
python -m scripts.verify_closed_form --jar
python -m scripts.train_ranker --rows 1000000 --test-profiles 20

# 4b. Predictor path: expand across frequency mixes, then train
python -m scripts.scenario_sweep --m 50 --out Output/scenario_ml.parquet
python -m scripts.validate_scenario_dataset
python -m scripts.train_predictor --parquet Output/scenario_ml.parquet
```

Trained bundles land in `Output/models/*.pkl` and appear automatically in the Predict page's model selector. To ship a retrained model with the repo, compress it under GitHub's 100 MB limit with `python -m scripts.setup_models --compress` (writes `*.pkl.xz`) and commit the `.xz`.

## How the ranker works (short version)

The Cost JAR's totals are linear in the query frequencies: `time = U_t·f`, `env = U_e·f + storage_overhead`, `fin = U_f·f + server_overhead`, where `U` are per-query *unit* costs. `verify_closed_form.py` proves this against the real JAR. The ranker therefore predicts only the frequency-independent unit costs and overheads from structural features (18 schema dims + volume + servers + model id); any workload is then applied analytically, so ranking all 135 models for a new config takes well under a second.

## Notes & gotchas

- Large derived artifacts — the 2.8 GB sweep dataset (`*.parquet`), sweep archives, run snapshots, and the 147 MB MSLE bake-off model — are **not** committed (regenerable from the scripts above). The **winner ranker and legacy predictor ARE** shipped, LZMA-compressed (`Output/models/*.pkl.xz`, decompressed on demand), so Predict works on a fresh clone; the paper's small figure-data CSVs under `report/` ship too, so `python -m report.make_figures` reproduces every figure without the dataset.
- `CostCalculation.jar` reads its inputs from `conf/`, not `Input/`. Always switch profiles through the UI or `swap_profile.py` — never by editing only one directory — or Phase 2 silently uses stale inputs.
- The Phase-2 JAR is resumable: if it times out or is killed, re-running it continues from the models that already have `costs.txt`.
- `Output/GeneratedDMs/` in the repo contains one complete example run (135 models + costs) so the dashboard works out of the box.

## Author

manpatell
