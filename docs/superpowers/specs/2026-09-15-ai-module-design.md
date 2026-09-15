# AI Module (5 weeks) — Design Specification

**Date:** 2026-09-15
**Author:** Aayush Raj Regmi
**Status:** Approved for planning

---

## 1. Purpose

Build a five-week teaching module covering the ten topics in `AIModuleSyllabus.md`, authored as Jupyter
notebooks in the same style as the existing DCS 404 material and published to the existing MkDocs site.

The module is the **applied / production layer** on top of DCS 404. It is not a second pass over
algorithms. The cohort has already completed DCS 404 and has seen regression, logistic regression,
decision trees, KNN, regularization, evaluation metrics, and time series. This module teaches how a model
becomes a system: framing, data discipline, pipelines, tracking, operations, and deployment.

## 2. Audience and constraints

- **Audience:** the DCS 404 cohort. Assume working Python, pandas, matplotlib, and scikit-learn basics,
  plus familiarity with linear/logistic regression, trees, KNN, and the standard metric families.
- **Contact time:** 5 weeks, 2 sessions per week, approximately 2 hours per session — 10 sessions.
- **Mapping:** one syllabus topic per session, one notebook per session. Ten notebooks.
- **Environment:** local only. No Docker installs, no cloud accounts, no paid services. Everything runs
  inside a Python virtual environment on a student laptop.

## 3. Module arc

| Week | # | Notebook | Syllabus topic | Treatment |
|------|---|----------|----------------|-----------|
| 1 | 01 | `01_problem_framing.ipynb` | ML Problem Framing & Industry Workflow | New material. Business objective to ML objective, problem types, when not to use ML, baseline-first thinking, success criteria, the ML lifecycle as a written project charter. |
| 1 | 02 | `02_data_preparation.ipynb` | Data Preparation for ML | Cleaning, missing values, categorical encoding, scaling, train/test split. Centrepiece is **data leakage**, taught as a diagnosis exercise on a planted leak. |
| 2 | 03 | `03_feature_engineering.ipynb` | Feature Engineering | Feature creation, selection, transformations, and class imbalance (`class_weight` vs resampling vs threshold shifting). |
| 2 | 04 | `04_sklearn_models.ipynb` | Building ML Models with Scikit-learn | Reframed, not re-taught. The estimator contract (`fit`/`predict`/`transform`), estimator selection, and how the algorithms from DCS 404 appear behind one uniform API. Cross-links to DCS 404 notebooks 03–09 for derivations. |
| 3 | 05 | `05_evaluation_metrics.ipynb` | Model Evaluation & Performance Metrics | Reframed. Metric **selection under business cost**, not metric definition. Cost matrices, operating points, and picking the number the project is actually judged on. DCS 404 notebook 08 carries the formulas. |
| 3 | 06 | `06_tuning_error_analysis.ipynb` | Model Tuning & Error Analysis | Grid, random and halving search; cross-validation done correctly; learning curves for over/underfitting; **slice-based error analysis** to find where the model fails and why. |
| 4 | 07 | `07_pipelines_tracking.ipynb` | ML Pipelines & Experiment Tracking | `Pipeline` + `ColumnTransformer` covering the whole preprocessing chain, reproducibility, MLflow local tracking, run comparison, model versioning. |
| 4 | 08 | `08_mlops_intro.ipynb` | Introduction to MLOps | Lifecycle and model management, deployment workflow, monitoring, **data and concept drift demonstrated on the spine dataset's shifted final quarter**, CI/CD concepts (read-only artifacts). |
| 5 | 09 | `09_deployment_apis.ipynb` | Model Deployment & APIs | Model persistence with `joblib`, the inference contract, a FastAPI prediction service, a Streamlit interface, input validation, and latency measurement. |
| 5 | 10 | `10_end_to_end_project.ipynb` | End-to-End ML Project | A complete run from problem statement to served model on a fresh problem, plus the student project brief and grading rubric. |

## 4. Spine dataset

All ten notebooks work the **same problem**, so the final week is a capstone rather than a fresh start.

The dataset is a generated Nepali SME loan-default dataset of roughly 12,000 rows, produced by a committed,
seeded script at `scripts/make_spine_data.py`. The existing `live-class/data/loans.csv` (300 rows) is too
small to support cross-validation, hyperparameter search, slice analysis, and drift demonstration.

Generation gives each syllabus topic a concrete hook:

| Planted property | Used by |
|------------------|---------|
| Missing values with a non-random mechanism | Notebook 02 |
| A leakage column (`recovery_agent_assigned`), recorded after default, highly predictive | Notebook 02 |
| High-cardinality categorical (`branch`) and skewed monetary columns | Notebook 03 |
| Roughly 12% positive rate — genuine class imbalance | Notebooks 03, 05 |
| `application_date` supporting a time-based split | Notebooks 02, 06 |
| Distribution shift in the final quarter | Notebook 08 |
| Asymmetric misclassification cost (missed default costs far more than a false alarm) | Notebook 05 |

Columns (indicative): `loan_id`, `application_date`, `branch`, `sector`, `loan_amount`, `tenure_months`,
`interest_rate`, `annual_income`, `credit_score`, `has_collateral`, `previous_loans`, `days_past_due_history`,
`employment_type`, `recovery_agent_assigned` (leak), `defaulted` (target).

Small supporting datasets appear only where a contrast needs them — for example a clean toy set when a
point is easier to see without noise.

## 5. Repository layout

```
notebooks/
  ai-module/
    01_problem_framing.ipynb
    ...
    10_end_to_end_project.ipynb
    data/                       spine dataset (CSV)
    resources/images/           hand-made diagrams and figures
scripts/
  make_spine_data.py            seeded generator for the spine dataset
app/
  api/main.py                   FastAPI prediction service (notebooks 09, 10)
  ui/app.py                     Streamlit interface (notebooks 09, 10)
  models/                       joblib artifacts
docs/
  AIModule/                     nbconvert output, mirrors docs/DCS404/
requirements-aimodule.txt       additional dependencies for this module
```

Nothing under `notebooks/` at the top level moves, and `docs/DCS404/` is untouched. The existing
`convert-all` Makefile target keeps working exactly as it does now.

## 6. Build and publish pipeline

Identical in shape to the DCS 404 pipeline already in the repository.

- `Makefile` gains `convert-ai` (single notebook) and `convert-ai-all` (batch), reading from
  `notebooks/ai-module` and writing to `docs/AIModule`, mirroring the existing `convert` and `convert-all`
  targets. Existing targets are not modified.
- `mkdocs.yml` gains a navigation group titled **Applied ML in Production**, placed after the existing
  Machine Learning and AI and Time Series groups, listing the ten notebooks plus the project page.
- Publication remains `mkdocs gh-deploy` via `make deploy`. Site name, theme, MathJax configuration, and
  copyright are unchanged.

## 7. Technology stack

Local execution only.

| Purpose | Tool |
|---------|------|
| Modelling and pipelines | `scikit-learn` |
| Class imbalance | `imbalanced-learn` |
| Experiment tracking | `mlflow`, run locally via `mlflow ui` |
| Model persistence | `joblib` |
| Prediction API | `fastapi` + `uvicorn` |
| Interface | `streamlit` |

Docker and GitHub Actions are taught conceptually. A `Dockerfile` and a CI workflow file are included as
annotated artifacts that students read and discuss but are never required to run. No student needs an
account with any external service.

## 8. Notebook authoring standard

Every notebook follows the structure already established in DCS 404:

1. Title and one-paragraph framing of why the topic matters.
2. "How to work through this" — the run-then-read rhythm and the notebook's internal structure.
3. Learning objectives, stated as capabilities.
4. A setup cell importing libraries and loading the spine dataset.
5. Numbered sections: intuition first, mathematics only where it earns its place, runnable code, and
   visualisations that build geometric or operational intuition.
6. "Your turn" — hands-on exercises.
7. "If you remember nothing else" — the compressed takeaway.
8. Further reading and a glossary.

Two additions specific to this module:

- **Industry note** callouts: short asides on how the technique is used, abused, or skipped in practice.
- **Back-link callouts** to the relevant DCS 404 page whenever a topic is being reframed rather than
  introduced, so students know exactly where the underlying derivation lives.

All notebooks are executed before conversion so that figures are present in the published Markdown.

## 9. Assessment

The final project is graded and completed individually. The project brief and rubric live in notebook 10
and are published to `docs/AIModule/`. The student project uses a dataset the student selects, not the
spine dataset, so the taught solution cannot be submitted directly.

Weighting between the final project and in-class work is set by the instructor at delivery time and is not
fixed by this specification.

## 10. Relationship to existing material

- `notebooks/` (DCS 404 notebooks 01–12 and `project/`) is unchanged.
- `docs/DCS404/` is unchanged.
- `live-class/` continues to hold live-session scratch notebooks and the Nepali business datasets. It is
  not part of the published site and is not restructured by this work.
- The new module cross-links into DCS 404 pages rather than duplicating their content.

## 11. Out of scope

- Deep learning, neural networks, and any GPU-dependent work.
- Cloud deployment, container orchestration, and managed ML platforms.
- Rewriting or reorganising DCS 404 material.
- Automated CI that builds or deploys the site; publication stays a manual `make deploy`.
