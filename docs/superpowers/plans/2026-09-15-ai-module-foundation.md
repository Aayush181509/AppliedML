# AI Module Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reproducible environment, spine dataset, build pipeline, and site navigation so that the ten AI-module teaching notebooks can be authored and published without any further infrastructure work.

**Architecture:** A seeded Python generator produces a single ~12,000-row loan-default CSV with deliberately planted teaching flaws (leakage column, non-random missingness, class imbalance, distribution drift in the final quarter). Notebooks live in `notebooks/ai-module/`, convert to Markdown with `nbconvert` through new `Makefile` targets mirroring the existing DCS 404 targets, and publish to `docs/AIModule/` under a new MkDocs navigation group. Nothing in the existing DCS 404 pipeline is modified.

**Tech Stack:** Python 3.12, `uv` for environment management, `numpy`/`pandas` for generation, `pytest` for dataset property tests, `jupyter`/`nbconvert` for conversion, `mkdocs` with the `readthedocs` theme for publication.

**Spec:** `docs/superpowers/specs/2026-09-15-ai-module-design.md`

## Global Constraints

- Python version is **3.12** — the existing DCS 404 notebooks were executed on 3.12.13, and `mlflow`/`streamlit` do not reliably support 3.14. Do not use the system `python3` (3.14.6).
- All dependency management goes through `uv`. It is already installed at `/opt/homebrew/bin/uv`.
- Random seed for all data generation is **42**, set explicitly. Generation must be deterministic: running the script twice produces byte-identical output.
- Local execution only. No Docker, no cloud services, no accounts, no network calls at runtime.
- The spine dataset target column is `defaulted`; the planted leakage column is `recovery_agent_assigned`; the date column is `application_date`.
- Do not modify `notebooks/*.ipynb` (DCS 404), `docs/DCS404/`, or the existing `convert`, `convert-all`, and `convert-project` Makefile targets.
- `live-class/` is gitignored — never add files there expecting them to be committed.
- Existing MkDocs settings (site name, `readthedocs` theme, MathJax config, copyright) stay as they are.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `requirements-aimodule.txt` | Pinned dependency list for the whole module |
| `scripts/make_spine_data.py` | Seeded generator producing the spine dataset; single responsibility, importable functions plus a CLI entry point |
| `tests/test_spine_data.py` | Property tests asserting every planted teaching flaw is actually present |
| `notebooks/ai-module/data/loan_default.csv` | The generated spine dataset, committed so students never need to run the generator |
| `notebooks/ai-module/00_setup_and_orientation.ipynb` | Student-facing setup check and module orientation; also the end-to-end proof of the build pipeline |
| `Makefile` | Gains `convert-ai`, `convert-ai-all`; existing targets untouched |
| `mkdocs.yml` | Gains the "Applied ML in Production" navigation group |
| `docs/AIModule/index.md` | Hand-written module landing page |
| `.gitignore` | Gains `mlruns/` and model-artifact rules |

---

### Task 1: Reproducible Python 3.12 environment

**Files:**
- Create: `requirements-aimodule.txt`
- Modify: `.gitignore` (append environment and artifact rules)

**Interfaces:**
- Consumes: nothing.
- Produces: a working virtual environment at `.venv/` whose interpreter is Python 3.12 and in which `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `mlflow`, `fastapi`, `uvicorn`, `streamlit`, `joblib`, `pytest`, `jupyter`, and `nbconvert` all import. Every later task runs commands via `.venv/bin/<tool>` or after `source .venv/bin/activate`.

- [ ] **Step 1: Write the dependency file**

Create `requirements-aimodule.txt`:

```
# AI Module — Applied ML in Production
# Python 3.12 required (mlflow and streamlit lag on 3.13+)

# Core data stack
numpy==2.1.3
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2

# Modelling
scikit-learn==1.5.2
imbalanced-learn==0.12.4

# Experiment tracking
mlflow==2.17.2

# Serving
fastapi==0.115.5
uvicorn==0.32.1
pydantic==2.10.2
streamlit==1.40.2
joblib==1.4.2

# Authoring and publication
jupyter==1.1.1
nbconvert==7.16.4
ipykernel==6.29.5
mkdocs==1.6.1

# Testing
pytest==8.3.4
```

- [ ] **Step 2: Create the environment and verify it fails before install**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
uv venv --python 3.12 .venv
.venv/bin/python -c "import pandas"
```

Expected: the venv is created and reports Python 3.12.x, then the import FAILS with `ModuleNotFoundError: No module named 'pandas'`. This confirms you are in a clean environment and not accidentally inheriting a global install.

- [ ] **Step 3: Install the dependencies**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
uv pip install --python .venv/bin/python -r requirements-aimodule.txt
```

If any pin fails to resolve on Python 3.12, relax that single pin to `>=` on the same major/minor line, note the change in the commit message, and keep every other pin exact.

- [ ] **Step 4: Verify every dependency imports**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/python -c "
import sys
assert sys.version_info[:2] == (3, 12), f'wrong python: {sys.version}'
mods = ['numpy','pandas','matplotlib','seaborn','sklearn','imblearn','mlflow',
        'fastapi','uvicorn','streamlit','joblib','pytest','nbconvert','mkdocs']
import importlib
for m in mods:
    importlib.import_module(m)
    print('ok', m)
print('ALL IMPORTS OK')
"
```

Expected: `ok` for all fourteen modules, then `ALL IMPORTS OK`. No traceback.

- [ ] **Step 5: Append ignore rules**

Append to `.gitignore`:

```
# AI Module
.venv/
mlruns/
mlartifacts/
app/models/*.joblib
app/models/*.pkl
.ipynb_checkpoints/
```

- [ ] **Step 6: Confirm the venv is ignored**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git status --porcelain | grep -c '^?? \.venv' || echo "venv correctly ignored"
```

Expected: `venv correctly ignored`. If `.venv/` appears in `git status`, the ignore rule did not take — fix it before committing.

- [ ] **Step 7: Commit**

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git add requirements-aimodule.txt .gitignore
git commit -m "build: pin AI module dependencies on Python 3.12

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Spine dataset generator — schema and determinism

**Files:**
- Create: `scripts/make_spine_data.py`
- Create: `tests/test_spine_data.py`

**Interfaces:**
- Consumes: the environment from Task 1.
- Produces: `generate_loans(n_rows: int = 12000, seed: int = 42) -> pandas.DataFrame`, returning a DataFrame with exactly these fifteen columns in this order: `loan_id`, `application_date`, `branch`, `sector`, `employment_type`, `loan_amount`, `tenure_months`, `interest_rate`, `annual_income`, `credit_score`, `has_collateral`, `previous_loans`, `days_past_due_history`, `recovery_agent_assigned`, `defaulted`. Tasks 3 and 4 extend this same function; Task 5 writes its output to disk.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spine_data.py`:

```python
"""Property tests for the AI module spine dataset.

Each test pins one teaching property that a notebook depends on. If a test
here fails, a notebook exercise somewhere stops working.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_spine_data import generate_loans  # noqa: E402

EXPECTED_COLUMNS = [
    "loan_id",
    "application_date",
    "branch",
    "sector",
    "employment_type",
    "loan_amount",
    "tenure_months",
    "interest_rate",
    "annual_income",
    "credit_score",
    "has_collateral",
    "previous_loans",
    "days_past_due_history",
    "recovery_agent_assigned",
    "defaulted",
]


@pytest.fixture(scope="module")
def df():
    return generate_loans(n_rows=12000, seed=42)


def test_row_count(df):
    assert len(df) == 12000


def test_columns_exact_and_ordered(df):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_loan_id_unique(df):
    assert df["loan_id"].is_unique


def test_application_date_is_datetime_and_in_range(df):
    dates = pd.to_datetime(df["application_date"])
    assert dates.min() >= pd.Timestamp("2023-01-01")
    assert dates.max() <= pd.Timestamp("2024-12-31")


def test_generation_is_deterministic():
    a = generate_loans(n_rows=500, seed=42)
    b = generate_loans(n_rows=500, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = generate_loans(n_rows=500, seed=42)
    b = generate_loans(n_rows=500, seed=7)
    assert not a.equals(b)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/pytest tests/test_spine_data.py -v
```

Expected: collection FAILS with `ModuleNotFoundError: No module named 'make_spine_data'`.

- [ ] **Step 3: Write the generator**

Create `scripts/make_spine_data.py`:

```python
"""Generate the AI module spine dataset: Nepali SME loan applications.

The dataset is deliberately imperfect. Every flaw is planted to support a
specific teaching moment, and `tests/test_spine_data.py` pins each one:

  * non-random missingness in `annual_income` and `credit_score`  (notebook 02)
  * `recovery_agent_assigned` — a leakage column recorded only after
    default has already happened                                   (notebook 02)
  * high-cardinality `branch`, skewed monetary columns             (notebook 03)
  * roughly 12 percent positive rate                               (notebooks 03, 05)
  * `application_date` supporting a time-ordered split             (notebooks 02, 06)
  * distribution and concept drift after 2024-10-01                (notebook 08)

Run:  python scripts/make_spine_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DEFAULT_ROWS = 12_000
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "ai-module"
    / "data"
    / "loan_default.csv"
)

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2024-12-31")
DRIFT_START = pd.Timestamp("2024-10-01")

SECTORS = ["Trade", "Service", "Manufacturing", "Agriculture", "Construction", "Tourism"]
SECTOR_WEIGHTS = [0.31, 0.24, 0.15, 0.14, 0.10, 0.06]

EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business Owner", "Informal"]
EMPLOYMENT_WEIGHTS = [0.34, 0.29, 0.22, 0.15]

CITIES = [
    "Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Biratnagar", "Birgunj",
    "Butwal", "Dharan", "Nepalgunj", "Janakpur", "Hetauda", "Itahari",
]

COLUMNS = [
    "loan_id",
    "application_date",
    "branch",
    "sector",
    "employment_type",
    "loan_amount",
    "tenure_months",
    "interest_rate",
    "annual_income",
    "credit_score",
    "has_collateral",
    "previous_loans",
    "days_past_due_history",
    "recovery_agent_assigned",
    "defaulted",
]


def _make_branches() -> list[str]:
    """Forty branch names across twelve cities — high-cardinality categorical."""
    branches: list[str] = []
    for city in CITIES:
        for n in range(1, 5):
            branches.append(f"{city}-{n:02d}")
    return branches[:40]


def generate_loans(n_rows: int = DEFAULT_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Return the spine dataset as a DataFrame.

    Deterministic: the same (n_rows, seed) always produces an identical frame.
    """
    rng = np.random.default_rng(seed)
    branches = _make_branches()

    span_days = (END_DATE - START_DATE).days
    offsets = rng.integers(0, span_days + 1, size=n_rows)
    application_date = START_DATE + pd.to_timedelta(offsets, unit="D")
    application_date = pd.Series(application_date).sort_values().reset_index(drop=True)

    df = pd.DataFrame(
        {
            "loan_id": [f"NL{200000 + i}" for i in range(n_rows)],
            "application_date": application_date,
            "branch": rng.choice(branches, size=n_rows),
            "sector": rng.choice(SECTORS, size=n_rows, p=SECTOR_WEIGHTS),
            "employment_type": rng.choice(
                EMPLOYMENT_TYPES, size=n_rows, p=EMPLOYMENT_WEIGHTS
            ),
        }
    )

    # Money columns are lognormal — right-skewed, as real loan books are.
    df["loan_amount"] = np.round(
        rng.lognormal(mean=12.6, sigma=0.75, size=n_rows) / 1000
    ) * 1000
    df["loan_amount"] = df["loan_amount"].clip(50_000, 10_000_000)

    df["tenure_months"] = rng.choice(
        [6, 12, 18, 24, 36, 48, 60], size=n_rows, p=[0.06, 0.20, 0.14, 0.24, 0.22, 0.09, 0.05]
    )

    df["annual_income"] = np.round(
        rng.lognormal(mean=13.1, sigma=0.62, size=n_rows) / 1000
    ) * 1000
    df["annual_income"] = df["annual_income"].clip(120_000, 40_000_000)

    df["credit_score"] = np.clip(
        rng.normal(loc=640, scale=95, size=n_rows).round(), 300, 850
    ).astype(int)

    df["previous_loans"] = rng.poisson(lam=1.3, size=n_rows).clip(0, 9)
    df["days_past_due_history"] = (
        rng.gamma(shape=0.9, scale=14.0, size=n_rows).round().clip(0, 180).astype(int)
    )
    df["has_collateral"] = np.where(rng.random(n_rows) < 0.58, "Yes", "No")

    # Interest rate rises with risk and falls with collateral.
    base_rate = 11.0
    risk_premium = (
        (700 - df["credit_score"]).clip(lower=0) / 100.0 * 1.6
        + df["days_past_due_history"] / 60.0
        - np.where(df["has_collateral"] == "Yes", 1.1, 0.0)
    )
    df["interest_rate"] = np.round(
        (base_rate + risk_premium + rng.normal(0, 0.45, n_rows)).clip(8.0, 22.0), 2
    )

    df = df[[c for c in COLUMNS if c in df.columns]]
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate_loans(n_rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df):,} rows x {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify the schema tests still fail**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/pytest tests/test_spine_data.py -v
```

Expected: `test_columns_exact_and_ordered` FAILS — `defaulted` and `recovery_agent_assigned` are not produced yet. The other five tests PASS. This is the correct intermediate state; Tasks 3 and 4 add the missing columns.

- [ ] **Step 5: Commit**

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git add scripts/make_spine_data.py tests/test_spine_data.py
git commit -m "feat: add spine dataset generator with applicant features

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Target variable, class imbalance, and concept drift

**Files:**
- Modify: `scripts/make_spine_data.py` (add default-probability logic inside `generate_loans`)
- Modify: `tests/test_spine_data.py` (append tests)

**Interfaces:**
- Consumes: `generate_loans(n_rows, seed)` from Task 2.
- Produces: a `defaulted` column of dtype `int` containing only `0` and `1`, with an overall positive rate between 0.10 and 0.14, and a positive rate on or after `2024-10-01` at least 1.6 times the rate before that date. Notebook 08 reads this drift; notebooks 03 and 05 read the imbalance.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spine_data.py`:

```python
def test_target_is_binary_int(df):
    assert df["defaulted"].dtype.kind == "i"
    assert set(df["defaulted"].unique()) == {0, 1}


def test_target_is_imbalanced_around_twelve_percent(df):
    rate = df["defaulted"].mean()
    assert 0.10 <= rate <= 0.14, f"default rate {rate:.3f} outside teaching range"


def test_default_rate_rises_with_risk(df):
    """Low credit scores must default more, or the problem is not learnable."""
    low = df[df["credit_score"] < 550]["defaulted"].mean()
    high = df[df["credit_score"] > 720]["defaulted"].mean()
    assert low > high * 2, f"signal too weak: low={low:.3f} high={high:.3f}"


def test_concept_drift_in_final_quarter(df):
    dates = pd.to_datetime(df["application_date"])
    before = df[dates < pd.Timestamp("2024-10-01")]["defaulted"].mean()
    after = df[dates >= pd.Timestamp("2024-10-01")]["defaulted"].mean()
    assert after >= before * 1.6, (
        f"drift too weak for notebook 08: before={before:.3f} after={after:.3f}"
    )


def test_data_drift_in_final_quarter(df):
    """Interest rates shift upward in the drift window — detectable by a
    distribution test, not only by the target rate."""
    dates = pd.to_datetime(df["application_date"])
    before = df[dates < pd.Timestamp("2024-10-01")]["interest_rate"].mean()
    after = df[dates >= pd.Timestamp("2024-10-01")]["interest_rate"].mean()
    assert after - before >= 0.8, f"rate shift {after - before:.2f} too small"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/pytest tests/test_spine_data.py -v -k "target or drift"
```

Expected: all five new tests FAIL with `KeyError: 'defaulted'`.

- [ ] **Step 3: Add the target and drift logic**

In `scripts/make_spine_data.py`, insert the following immediately after the `df["interest_rate"] = ...` block and before the `df = df[[c for c in COLUMNS if c in df.columns]]` line:

```python
    # ---- Drift window -----------------------------------------------------
    # From 2024-10-01 the lending environment deteriorates: rates rise,
    # incomes soften, and the same applicant profile defaults more often.
    # Notebook 08 detects this; nothing before notebook 08 should mention it.
    in_drift = df["application_date"] >= DRIFT_START
    df.loc[in_drift, "interest_rate"] = np.round(
        (df.loc[in_drift, "interest_rate"] + rng.normal(1.4, 0.3, in_drift.sum()))
        .clip(8.0, 24.0),
        2,
    )
    df.loc[in_drift, "annual_income"] = np.round(
        df.loc[in_drift, "annual_income"] * rng.normal(0.88, 0.05, in_drift.sum())
        / 1000
    ) * 1000

    # ---- Default probability ---------------------------------------------
    # A logistic function of genuine risk drivers, so the problem is learnable
    # but never perfectly separable.
    logit = (
        -1.15
        + (700 - df["credit_score"]) / 100.0 * 0.95
        + df["days_past_due_history"] / 45.0
        + np.log(df["loan_amount"] / df["annual_income"].clip(lower=1)) * 0.55
        + np.where(df["has_collateral"] == "No", 0.42, -0.30)
        + np.where(df["employment_type"] == "Informal", 0.55, 0.0)
        + np.where(df["sector"] == "Construction", 0.33, 0.0)
        + np.where(df["sector"] == "Tourism", 0.28, 0.0)
        + np.where(in_drift, 1.05, 0.0)
        + rng.normal(0, 0.35, n_rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    df["defaulted"] = (rng.random(n_rows) < probability).astype(int)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/pytest tests/test_spine_data.py -v
```

Expected: every test PASSES except `test_columns_exact_and_ordered`, which still fails because `recovery_agent_assigned` does not exist yet.

If `test_target_is_imbalanced_around_twelve_percent` fails, adjust only the `-1.15` intercept in the `logit` expression — lower it to reduce the default rate, raise it to increase — and re-run. Change nothing else.

- [ ] **Step 5: Commit**

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git add scripts/make_spine_data.py tests/test_spine_data.py
git commit -m "feat: add default target with imbalance and Q4 drift

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Planted leakage column and non-random missingness

**Files:**
- Modify: `scripts/make_spine_data.py`
- Modify: `tests/test_spine_data.py`

**Interfaces:**
- Consumes: the `defaulted` column from Task 3.
- Produces: `recovery_agent_assigned` (values `"Yes"`/`"No"`, assigned to roughly 88 percent of defaulters and 3 percent of non-defaulters) and `NaN` values in `annual_income` and `credit_score` following a non-random mechanism. After this task `generate_loans` returns all fifteen columns in the order declared by `COLUMNS`, which is what notebook 02 and every later notebook load.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spine_data.py`:

```python
def test_leakage_column_almost_perfectly_predicts_target(df):
    """`recovery_agent_assigned` is recorded after default. A student who
    leaves it in gets a suspiciously excellent model — that is the lesson."""
    assert set(df["recovery_agent_assigned"].unique()) == {"Yes", "No"}
    flagged = df[df["recovery_agent_assigned"] == "Yes"]["defaulted"].mean()
    clean = df[df["recovery_agent_assigned"] == "No"]["defaulted"].mean()
    assert flagged > 0.70, f"leak too weak to be seductive: {flagged:.3f}"
    assert clean < 0.06, f"non-flagged rows too noisy: {clean:.3f}"


def test_missingness_exists_and_is_teachable(df):
    income_missing = df["annual_income"].isna().mean()
    score_missing = df["credit_score"].isna().mean()
    assert 0.04 <= income_missing <= 0.14, f"income missingness {income_missing:.3f}"
    assert 0.03 <= score_missing <= 0.12, f"score missingness {score_missing:.3f}"


def test_income_missingness_is_not_random(df):
    """Informal workers are far likelier to have no documented income.
    Dropping those rows silently drops the highest-risk applicants."""
    informal = df[df["employment_type"] == "Informal"]["annual_income"].isna().mean()
    salaried = df[df["employment_type"] == "Salaried"]["annual_income"].isna().mean()
    assert informal > salaried * 3, (
        f"missingness looks random: informal={informal:.3f} salaried={salaried:.3f}"
    )


def test_credit_score_missingness_tracks_first_time_borrowers(df):
    first_time = df[df["previous_loans"] == 0]["credit_score"].isna().mean()
    repeat = df[df["previous_loans"] > 0]["credit_score"].isna().mean()
    assert first_time > repeat * 3, (
        f"missingness looks random: first={first_time:.3f} repeat={repeat:.3f}"
    )


def test_no_missing_values_in_target_or_id(df):
    assert df["defaulted"].notna().all()
    assert df["loan_id"].notna().all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/pytest tests/test_spine_data.py -v -k "leak or missing"
```

Expected: the leakage and missingness tests FAIL with `KeyError: 'recovery_agent_assigned'` and assertion failures showing zero missingness.

- [ ] **Step 3: Add the leak and the missingness**

In `scripts/make_spine_data.py`, insert immediately after the `df["defaulted"] = ...` line and before the `df = df[[c for c in COLUMNS if c in df.columns]]` line:

```python
    # ---- Planted leakage --------------------------------------------------
    # A recovery agent is assigned only *after* an account has gone bad, so
    # this column cannot exist at scoring time. It is in the file because real
    # extracts are assembled from whatever the warehouse happens to hold.
    assign_probability = np.where(df["defaulted"] == 1, 0.88, 0.03)
    df["recovery_agent_assigned"] = np.where(
        rng.random(n_rows) < assign_probability, "Yes", "No"
    )

    # ---- Non-random missingness ------------------------------------------
    # Income is undocumented for informal workers; credit scores do not exist
    # for first-time borrowers. Both groups are higher risk, so dropping
    # incomplete rows quietly removes the applicants that matter most.
    income_missing_probability = np.select(
        [
            df["employment_type"] == "Informal",
            df["employment_type"] == "Self-Employed",
        ],
        [0.34, 0.11],
        default=0.02,
    )
    df.loc[rng.random(n_rows) < income_missing_probability, "annual_income"] = np.nan

    score_missing_probability = np.where(df["previous_loans"] == 0, 0.22, 0.015)
    df["credit_score"] = df["credit_score"].astype("float64")
    df.loc[rng.random(n_rows) < score_missing_probability, "credit_score"] = np.nan
```

- [ ] **Step 4: Run the full test suite**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/pytest tests/test_spine_data.py -v
```

Expected: all sixteen tests PASS, including `test_columns_exact_and_ordered`.

If `test_missingness_exists_and_is_teachable` fails on the overall rate, adjust only the probabilities in `income_missing_probability` and `score_missing_probability`. If `test_leakage_column_almost_perfectly_predicts_target` fails, adjust only the `0.88` and `0.03` in `assign_probability`.

- [ ] **Step 5: Commit**

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git add scripts/make_spine_data.py tests/test_spine_data.py
git commit -m "feat: plant leakage column and non-random missingness

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Generate and commit the dataset

**Files:**
- Create: `notebooks/ai-module/data/loan_default.csv`
- Create: `notebooks/ai-module/data/README.md`

**Interfaces:**
- Consumes: `scripts/make_spine_data.py` from Tasks 2–4.
- Produces: a committed CSV at `notebooks/ai-module/data/loan_default.csv`. Every notebook loads it with `pd.read_csv("data/loan_default.csv", parse_dates=["application_date"])` relative to `notebooks/ai-module/`.

- [ ] **Step 1: Generate the dataset**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/python scripts/make_spine_data.py
```

Expected: `wrote 12,000 rows x 15 columns to .../notebooks/ai-module/data/loan_default.csv`

- [ ] **Step 2: Verify determinism by regenerating**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
shasum -a 256 notebooks/ai-module/data/loan_default.csv > /tmp/hash1.txt
.venv/bin/python scripts/make_spine_data.py
shasum -a 256 notebooks/ai-module/data/loan_default.csv > /tmp/hash2.txt
diff /tmp/hash1.txt /tmp/hash2.txt && echo "DETERMINISTIC"
```

Expected: `DETERMINISTIC`. If the hashes differ, an unseeded source of randomness has crept into `generate_loans` — find it and route it through `rng` before continuing.

- [ ] **Step 3: Sanity-check the file the way a student will see it**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('notebooks/ai-module/data/loan_default.csv', parse_dates=['application_date'])
print(df.shape)
print(df.dtypes)
print('default rate:', round(df['defaulted'].mean(), 4))
print('missing:'); print(df.isna().sum()[lambda s: s > 0])
print(df.head(3).to_string())
"
```

Expected: shape `(12000, 15)`, a default rate near 0.12, missing values only in `annual_income` and `credit_score`, and `application_date` parsed as `datetime64[ns]`.

- [ ] **Step 4: Write the data dictionary**

Create `notebooks/ai-module/data/README.md`:

```markdown
# Spine dataset — `loan_default.csv`

Simulated SME loan applications from a Nepali lender, 2023-01-01 to 2024-12-31.
Every notebook in this module works this one problem, so that the final week is
a capstone rather than a fresh start.

Regenerate with `python scripts/make_spine_data.py` (seeded — output is
identical every run).

| Column | Type | Meaning |
|--------|------|---------|
| `loan_id` | string | Unique application reference |
| `application_date` | date | When the application was submitted |
| `branch` | string | Originating branch, 40 distinct values |
| `sector` | string | Borrower's business sector |
| `employment_type` | string | Salaried, Self-Employed, Business Owner, Informal |
| `loan_amount` | float | Principal requested, NPR |
| `tenure_months` | int | Requested repayment period |
| `interest_rate` | float | Offered annual rate, percent |
| `annual_income` | float | Declared annual income, NPR — **has missing values** |
| `credit_score` | float | Bureau score 300–850 — **has missing values** |
| `has_collateral` | string | Yes / No |
| `previous_loans` | int | Count of prior loans with this lender |
| `days_past_due_history` | int | Worst historical delinquency, days |
| `recovery_agent_assigned` | string | Yes / No |
| `defaulted` | int | **Target.** 1 if the loan defaulted |

The data dictionary deliberately stops at the column descriptions. Explaining
that `recovery_agent_assigned` is target leakage, or that the missing values in
`annual_income` and `credit_score` are not missing at random, would give away
the exercises in notebook 02. Those explanations belong in the notebooks, after
the learner has found the problem.
```

- [ ] **Step 5: Confirm the CSV is not blocked by an ignore rule**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git check-ignore -v notebooks/ai-module/data/loan_default.csv && echo "BLOCKED — fix .gitignore" || echo "trackable"
```

Expected: `trackable`.

- [ ] **Step 6: Commit**

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git add notebooks/ai-module/data/loan_default.csv notebooks/ai-module/data/README.md
git commit -m "feat: add generated spine dataset and data dictionary

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Build pipeline — Makefile targets and MkDocs navigation

**Files:**
- Modify: `Makefile` (append new targets; existing targets untouched)
- Modify: `mkdocs.yml:5-24` (append a navigation group after the Final Project entry)
- Create: `docs/AIModule/index.md`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `make convert-ai NOTEBOOK=<path>` converting a single notebook, and `make convert-ai-all` converting every notebook in `notebooks/ai-module/` into `docs/AIModule/`. Task 7 runs both.

- [ ] **Step 1: Append the Makefile targets**

At the top of `Makefile`, after the existing `OUTPUT_DIR := docs/DCS404` line, add:

```makefile
# AI Module (Applied ML in Production)
AI_INPUT_DIR  := notebooks/ai-module
AI_OUTPUT_DIR := docs/AIModule
AI_NOTEBOOK   ?= notebooks/ai-module/00_setup_and_orientation.ipynb
AI_NOTEBOOKS  := $(wildcard $(AI_INPUT_DIR)/*.ipynb)
```

Add `convert-ai convert-ai-all` to the existing `.PHONY` line, and append these targets at the end of the file:

```makefile
# Single AI-module notebook: make convert-ai AI_NOTEBOOK=notebooks/ai-module/01_problem_framing.ipynb
convert-ai:
	@mkdir -p $(AI_OUTPUT_DIR)
	jupyter nbconvert --to markdown $(AI_NOTEBOOK) --output-dir=$(AI_OUTPUT_DIR)

# All AI-module notebooks
convert-ai-all:
	@echo "Converting all notebooks in $(AI_INPUT_DIR)/ to Markdown..."
	@mkdir -p $(AI_OUTPUT_DIR)
	@for nb in $(AI_NOTEBOOKS); do \
		echo "Converting $$nb..."; \
		jupyter nbconvert --to markdown $$nb --output-dir=$(AI_OUTPUT_DIR); \
	done
```

- [ ] **Step 2: Verify the existing targets are unchanged**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git diff Makefile
```

Expected: the diff shows only additions. No line inside the existing `convert`, `convert-all`, `convert-project`, `build`, `serve`, or `deploy` recipes is modified.

- [ ] **Step 3: Write the module landing page**

Create `docs/AIModule/index.md`:

~~~markdown
# Applied ML in Production

A five-week module on what happens to a model *after* the algorithm works.

DCS 404 covered how the algorithms learn. This module covers everything that
stands between a notebook that fits a model and a system somebody depends on:
framing the problem so the answer is worth having, preparing data without
poisoning it, engineering features, selecting a metric the business actually
cares about, tuning honestly, wrapping the whole chain in a reproducible
pipeline, tracking experiments, monitoring for drift, and serving predictions
over an API.

## How the module runs

Ten sessions over five weeks, two sessions per week, one notebook per session.

Every notebook works the **same problem**: predicting default on SME loan
applications from a Nepali lender. By the final week you are not starting
something new — you are finishing what you have been building since week one.

## What you need

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-aimodule.txt
```

Everything runs locally. No cloud accounts, no Docker, no paid services.

## Prerequisites

This module assumes DCS 404. Where a topic is being *reframed* rather than
introduced, the notebook links back to the DCS 404 page holding the derivation.
~~~

- [ ] **Step 4: Add the navigation group**

In `mkdocs.yml`, immediately after the existing `Final Project` group (the `- Project Work: DCS404/project/00_final_project.md` line) and before the `theme:` key, add:

```yaml
  - Applied ML in Production:
    - Module Overview: AIModule/index.md
    - Setup and Orientation: AIModule/00_setup_and_orientation.md
```

Leave the `nav` entries for the DCS 404 groups exactly as they are. The remaining ten notebook entries are added in the content plans as each notebook is authored.

- [ ] **Step 5: Verify the site still builds**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
.venv/bin/mkdocs build 2>&1 | tail -20
```

Expected: the build completes. A `WARNING` about `AIModule/00_setup_and_orientation.md` not existing is correct at this point — Task 7 creates it. There must be no `ERROR` and no `Aborted`.

- [ ] **Step 6: Commit**

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git add Makefile mkdocs.yml docs/AIModule/index.md
git commit -m "build: add AI module conversion targets and site navigation

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Orientation notebook and end-to-end pipeline proof

**Files:**
- Create: `notebooks/ai-module/00_setup_and_orientation.ipynb`
- Create: `docs/AIModule/00_setup_and_orientation.md` (generated by `make convert-ai`)

**Interfaces:**
- Consumes: the environment (Task 1), the dataset (Task 5), and the build targets (Task 6).
- Produces: a published page proving notebook → Markdown → site works end to end. Every subsequent notebook follows this file's cell structure and its dataset-loading cell verbatim.

- [ ] **Step 1: Author the notebook**

Create `notebooks/ai-module/00_setup_and_orientation.ipynb` with a Python 3 kernel and these cells, in order. Follow the DCS 404 house style: prose that explains *why* before *what*, and no cell that is only a code dump.

**Cell 1 (markdown)** — title and framing:

```markdown
# Setup and Orientation

**Applied ML in Production · Session 0**

---

This module is about the distance between a model that works and a model that
is *used*. That distance is where most machine learning projects die, and
almost none of it is algorithmic.

Over five weeks we will take one problem — predicting default on SME loan
applications — from a vague business complaint all the way to a running
prediction service. The same dataset, the same problem, ten sessions. Nothing
gets thrown away and restarted.

This session does two things: it confirms your environment works, and it
introduces the data you will be living with.
```

**Cell 2 (markdown)** — how to work through the module:

```markdown
## How to work through this

Run every code cell (`Shift + Enter`), read the output, *then* read the
commentary. Cells build on each other, so if something errors, run from the top.

Each notebook ends with exercises and a short "if you remember nothing else"
summary. The exercises are where the learning actually happens — the code cells
are just the demonstration.
```

**Cell 3 (markdown)** — learning objectives:

```markdown
## Learning objectives

After this session you will be able to:

- Set up and verify the module's Python environment.
- Load the spine dataset and describe what each column means.
- State the business problem this module solves, in one sentence.
- Explain why the module uses a single dataset throughout rather than a new one
  each week.
```

**Cell 4 (markdown)** — setup instructions:

~~~markdown
## Environment

From the repository root, once:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-aimodule.txt
```

Then select `.venv` as this notebook's kernel. The next cell checks that
everything the module needs is present.
~~~

**Cell 5 (code)** — environment check:

```python
import sys
import importlib

REQUIRED = [
    "numpy", "pandas", "matplotlib", "seaborn", "sklearn", "imblearn",
    "mlflow", "fastapi", "streamlit", "joblib",
]

print(f"Python {sys.version.split()[0]}")
if sys.version_info[:2] != (3, 12):
    print("  WARNING: this module is built and tested on Python 3.12")

missing = []
for name in REQUIRED:
    try:
        module = importlib.import_module(name)
        print(f"  ok   {name:<12} {getattr(module, '__version__', '')}")
    except ImportError:
        missing.append(name)
        print(f"  MISSING  {name}")

if missing:
    print(f"\nInstall the missing packages before continuing: {', '.join(missing)}")
else:
    print("\nEnvironment ready.")
```

**Cell 6 (markdown)** — the problem statement:

```markdown
## The problem

A lender issues working-capital loans to small and medium businesses across
Nepal. Credit officers assess each application by hand. Volume has grown faster
than the credit team, decisions have become inconsistent between branches, and
the default rate has drifted upward.

The request that arrives on your desk is: *"can we use machine learning to
predict which loans will default?"*

That sentence is not yet a machine learning problem. Turning it into one — with
a defined target, a defined population, a defined decision, and a defined
measure of success — is the entire content of session 1.
```

**Cell 7 (code)** — load the data:

```python
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/loan_default.csv")

loans = pd.read_csv(DATA_PATH, parse_dates=["application_date"])

print(f"{len(loans):,} applications, {loans.shape[1]} columns")
print(f"{loans['application_date'].min():%Y-%m-%d} to {loans['application_date'].max():%Y-%m-%d}")
loans.head()
```

**Cell 8 (code)** — first look:

```python
print("Column types and completeness\n")
summary = pd.DataFrame({
    "dtype": loans.dtypes.astype(str),
    "missing": loans.isna().sum(),
    "missing_pct": (loans.isna().mean() * 100).round(1),
    "unique": loans.nunique(),
})
print(summary.to_string())

print(f"\nDefault rate: {loans['defaulted'].mean():.1%}")
```

**Cell 9 (markdown)** — what the first look already tells us:

```markdown
### Three things that output already told you

**The classes are imbalanced.** Roughly one loan in eight defaults. A model that
predicts "never defaults" for every application is right about 88 percent of the
time and completely useless. Accuracy is already a trap, and we have not written
a line of modelling code.

**Two columns have missing values.** `annual_income` and `credit_score`. The
convenient move is `dropna()`. Session 2 shows what that quietly costs you.

**One column is not what it appears to be.** We will not say which one yet.
Session 2 is about finding it.

Do not fix any of this now. Each problem is the subject of a later session, and
seeing it bite before seeing it fixed is the point.
```

**Cell 10 (markdown)** — the module map:

```markdown
## The ten sessions

| Week | Session | Topic |
|------|---------|-------|
| 1 | 1 | ML problem framing and the industry workflow |
| 1 | 2 | Data preparation — and the leakage hunt |
| 2 | 3 | Feature engineering and class imbalance |
| 2 | 4 | Building models with scikit-learn |
| 3 | 5 | Evaluation metrics and the cost of being wrong |
| 3 | 6 | Tuning and error analysis |
| 4 | 7 | Pipelines and experiment tracking |
| 4 | 8 | Introduction to MLOps |
| 5 | 9 | Deployment and APIs |
| 5 | 10 | End-to-end project |

Sessions 4, 5 and 6 revisit ground DCS 404 covered. They are not a repeat — the
question there was *how does this algorithm work*, and the question here is
*how do I choose, judge, and defend one*. Where a derivation is needed, the
notebook links back to the DCS 404 page rather than repeating it.
```

**Cell 11 (markdown)** — closing:

```markdown
## If you remember nothing else

One problem, ten sessions, one dataset. The messiness you just saw in the data
summary is not an accident and it is not something to clean up before the real
work starts — it *is* the real work.

## Your turn

1. Open `data/README.md` and read the column descriptions. Which column would
   you be suspicious of, and why?
2. Compute the default rate separately for each `sector`. Which sector looks
   riskiest?
3. Write one sentence stating what decision this model would actually change.
   Keep it — session 1 opens with it.
```

- [ ] **Step 2: Execute the notebook to populate outputs**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule/notebooks/ai-module
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 00_setup_and_orientation.ipynb
```

Expected: completes with no error. If the data path fails, confirm you are running from `notebooks/ai-module/` — the notebook loads `data/loan_default.csv` relative to its own directory, and every later notebook does the same.

- [ ] **Step 3: Convert to Markdown**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
PATH=".venv/bin:$PATH" make convert-ai
```

Expected: `docs/AIModule/00_setup_and_orientation.md` is written.

- [ ] **Step 4: Verify the published output**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
test -f docs/AIModule/00_setup_and_orientation.md && echo "page exists"
grep -c "Default rate" docs/AIModule/00_setup_and_orientation.md
.venv/bin/mkdocs build 2>&1 | tail -20
```

Expected: `page exists`, a non-zero grep count confirming executed output was captured, and a clean `mkdocs build` with no `ERROR` and no warning about a missing `AIModule/` page.

- [ ] **Step 5: Verify the DCS 404 pipeline is undamaged**

Run:

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git status --porcelain docs/DCS404 notebooks/*.ipynb
```

Expected: no output. If any DCS 404 file shows as modified, revert it — this plan must not touch that material.

- [ ] **Step 6: Commit**

```bash
cd /Users/aayush/Documents/Kings/AI/AIModule
git add notebooks/ai-module/00_setup_and_orientation.ipynb docs/AIModule/00_setup_and_orientation.md
git commit -m "feat: add setup and orientation notebook

Proves the notebook to Markdown to site pipeline end to end.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `.venv/bin/pytest tests/test_spine_data.py` passes all sixteen tests.
- `python scripts/make_spine_data.py` run twice produces an identical file hash.
- `make convert-ai-all` converts every notebook in `notebooks/ai-module/` into `docs/AIModule/`.
- `mkdocs build` completes with no errors and the new navigation group renders.
- `git status --porcelain docs/DCS404 notebooks/*.ipynb` is empty.

## Follow-on plans

- **Plan B — notebooks 01–06** (weeks 1–3): framing, data preparation, feature engineering, scikit-learn, evaluation metrics, tuning and error analysis.
- **Plan C — notebooks 07–10 and the service** (weeks 4–5): pipelines and tracking, MLOps, deployment with FastAPI and Streamlit under `app/`, and the end-to-end project brief and rubric.

Each is written after the preceding plan lands, so the authoring conventions established here are already fixed.
