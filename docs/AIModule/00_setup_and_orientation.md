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

## How to work through this

Run every code cell (`Shift + Enter`), read the output, *then* read the
commentary. Cells build on each other, so if something errors, run from the top.

Each notebook ends with exercises and a short "if you remember nothing else"
summary. The exercises are where the learning actually happens — the code cells
are just the demonstration.

## Learning objectives

After this session you will be able to:

- Set up and verify the module's Python environment.
- Load the spine dataset and describe what each column means.
- State the business problem this module solves, in one sentence.
- Explain why the module uses a single dataset throughout rather than a new one
  each week.

## Environment

From the repository root, once:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-aimodule.txt
```

Then select `.venv` as this notebook's kernel. The next cell checks that
everything the module needs is present.


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

    Python 3.12.13
      ok   numpy        2.1.3


      ok   pandas       2.2.3
      ok   matplotlib   3.9.2


      ok   seaborn      0.13.2
      ok   sklearn      1.5.2


      ok   imblearn     0.12.4


      ok   mlflow       2.17.2
      ok   fastapi      0.115.5
      ok   streamlit    1.40.2
      ok   joblib       1.4.2
    
    Environment ready.


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


```python
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/loan_default.csv")

loans = pd.read_csv(DATA_PATH, parse_dates=["application_date"])

print(f"{len(loans):,} applications, {loans.shape[1]} columns")
print(f"{loans['application_date'].min():%Y-%m-%d} to {loans['application_date'].max():%Y-%m-%d}")
loans.head()
```

    12,000 applications, 15 columns
    2023-01-01 to 2024-12-31





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_id</th>
      <th>application_date</th>
      <th>branch</th>
      <th>sector</th>
      <th>employment_type</th>
      <th>loan_amount</th>
      <th>tenure_months</th>
      <th>interest_rate</th>
      <th>annual_income</th>
      <th>credit_score</th>
      <th>has_collateral</th>
      <th>previous_loans</th>
      <th>days_past_due_history</th>
      <th>recovery_agent_assigned</th>
      <th>defaulted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NL200000</td>
      <td>2023-01-01</td>
      <td>Birgunj-01</td>
      <td>Service</td>
      <td>Business Owner</td>
      <td>417000.0</td>
      <td>36</td>
      <td>10.28</td>
      <td>194000.0</td>
      <td>652.0</td>
      <td>Yes</td>
      <td>2</td>
      <td>40</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NL200001</td>
      <td>2023-01-01</td>
      <td>Biratnagar-03</td>
      <td>Service</td>
      <td>Self-Employed</td>
      <td>154000.0</td>
      <td>12</td>
      <td>9.82</td>
      <td>NaN</td>
      <td>695.0</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NL200002</td>
      <td>2023-01-01</td>
      <td>Kathmandu-03</td>
      <td>Trade</td>
      <td>Salaried</td>
      <td>163000.0</td>
      <td>24</td>
      <td>13.04</td>
      <td>948000.0</td>
      <td>565.0</td>
      <td>No</td>
      <td>2</td>
      <td>22</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NL200003</td>
      <td>2023-01-01</td>
      <td>Lalitpur-01</td>
      <td>Tourism</td>
      <td>Salaried</td>
      <td>634000.0</td>
      <td>18</td>
      <td>12.98</td>
      <td>327000.0</td>
      <td>505.0</td>
      <td>Yes</td>
      <td>2</td>
      <td>21</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NL200004</td>
      <td>2023-01-01</td>
      <td>Birgunj-04</td>
      <td>Trade</td>
      <td>Salaried</td>
      <td>196000.0</td>
      <td>60</td>
      <td>10.08</td>
      <td>235000.0</td>
      <td>643.0</td>
      <td>Yes</td>
      <td>3</td>
      <td>2</td>
      <td>No</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




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

    Column types and completeness
    
                                      dtype  missing  missing_pct  unique
    loan_id                          object        0          0.0   12000
    application_date         datetime64[ns]        0          0.0     731
    branch                           object        0          0.0      40
    sector                           object        0          0.0       6
    employment_type                  object        0          0.0       4
    loan_amount                     float64        0          0.0    1362
    tenure_months                     int64        0          0.0       7
    interest_rate                   float64        0          0.0     778
    annual_income                   float64     1122          9.4    1613
    credit_score                    float64      851          7.1     493
    has_collateral                   object        0          0.0       2
    previous_loans                    int64        0          0.0       9
    days_past_due_history             int64        0          0.0      99
    recovery_agent_assigned          object        0          0.0       2
    defaulted                         int64        0          0.0       2
    
    Default rate: 13.5%


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
