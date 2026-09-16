# Model Deployment and APIs

**Applied ML in Production · Session 9**

---

Eight sessions of work exist only inside a notebook. A credit officer cannot open
a notebook, and neither can the bank's loan origination system.

This session closes that gap. We save the pipeline to a file, wrap it in a small
web service that validates what it is sent, measure how fast it answers, and put a
simple interface in front of it. By the end this project is something another
program can call.

The engineering here is deliberately modest — a hundred lines in total. The ideas
that matter are the **inference contract** (what the service promises), **input
validation** (what it refuses), and the discipline that the code serving the model
must be the code that trained it.

## How to work through this

This notebook writes real files into the repository: `app/api/main.py`,
`app/ui/app.py`, and the saved model in `app/models/`. Run it and they appear,
ready to launch from a terminal.

Run each cell, read the output, then read the commentary. If a cell errors, run
from the top.

## Learning objectives

After this session you will be able to:

- Persist a fitted pipeline with `joblib`, and say what else must be saved with it.
- Write down an **inference contract**: inputs, outputs, errors, and version.
- Build a **FastAPI** prediction service with Pydantic validation.
- Test the service end to end and read a `422` validation error.
- Measure prediction **latency** and compare it to the charter's budget.
- Build a **Streamlit** interface for the people who will not call an API.
- List what a prediction service must log for session 8's monitoring to work.

## Setup

We train the final pipeline on the whole dataset — at deployment time you use
everything you have — and record the context that has to travel with it.


```python
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_repository_root():
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / "data").is_dir():
            return folder
    raise FileNotFoundError("could not find the repository root")


ROOT = find_repository_root()
loans = pd.read_csv(ROOT / "data" / "loan_default.csv", parse_dates=["application_date"])

NUMERIC = ["loan_amount", "tenure_months", "interest_rate", "previous_loans",
           "days_past_due_history", "annual_income", "credit_score"]
CATEGORICAL = ["sector", "employment_type", "has_collateral"]
THRESHOLD = 0.20            # the operating point agreed in session 5

model = Pipeline([
    ("prepare", ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", drop="first"))]),
         CATEGORICAL),
    ])),
    ("classify", LogisticRegression(max_iter=1000)),
])

model.fit(loans[NUMERIC + CATEGORICAL], loans["defaulted"])
print("trained on", f"{len(loans):,}", "applications")
```

    trained on 12,000 applications


---

## 1. Saving the model — and everything around it

`joblib.dump` writes the fitted pipeline to disk. Because it is a *pipeline*, the
file contains the imputers, the scaler, the encoder and the model together, so the
serving code cannot preprocess differently from the training code. That is session
7's payoff arriving.

But a model file alone is not deployable. Three questions it cannot answer:

- Which threshold turns this probability into a decision?
- What fields does it expect, and of what type?
- Which version of scikit-learn wrote it, and from which data?

So we save a small metadata file beside it. This is the cheapest habit in
production machine learning and it prevents the most confusing incidents.


```python
MODEL_DIR = ROOT / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "loan_default_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "loan_default_pipeline.json"

joblib.dump(model, MODEL_PATH)

data_file = ROOT / "data" / "loan_default.csv"
metadata = {
    "name": "loan-default",
    "version": "1.0.0",
    "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "training_rows": len(loans),
    "training_data_md5": hashlib.md5(data_file.read_bytes()).hexdigest()[:12],
    "numeric_features": NUMERIC,
    "categorical_features": CATEGORICAL,
    "threshold": THRESHOLD,
    "sklearn_version": sklearn.__version__,
    "notes": "Probability of default within 12 months, scored at application time.",
}
METADATA_PATH.write_text(json.dumps(metadata, indent=2))

print(f"model    {MODEL_PATH.name}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
print(f"metadata {METADATA_PATH.name}")
print(json.dumps(metadata, indent=2)[:400], "...")
```

    model    loan_default_pipeline.joblib  (5.5 KB)
    metadata loan_default_pipeline.json
    {
      "name": "loan-default",
      "version": "1.0.0",
      "trained_at": "2026-09-16 07:15",
      "training_rows": 12000,
      "training_data_md5": "18b4b7c4918f",
      "numeric_features": [
        "loan_amount",
        "tenure_months",
        "interest_rate",
        "previous_loans",
        "days_past_due_history",
        "annual_income",
        "credit_score"
      ],
      "categorical_features": [
        "sector",
        "employment_type",
        ...



```python
# The round trip that must always be checked: does the reloaded model agree?
reloaded = joblib.load(MODEL_PATH)

sample = loans[NUMERIC + CATEGORICAL].head(5)
original_predictions = model.predict_proba(sample)[:, 1]
reloaded_predictions = reloaded.predict_proba(sample)[:, 1]

print("original:", original_predictions.round(4))
print("reloaded:", reloaded_predictions.round(4))
print("identical:", np.allclose(original_predictions, reloaded_predictions))
```

    original: [0.0856 0.0229 0.1419 0.3178 0.0383]
    reloaded: [0.0856 0.0229 0.1419 0.3178 0.0383]
    identical: True


> **The version trap.** A `joblib` file is a pickled Python object graph. Load it
> with a different scikit-learn version and you may get a warning, a subtly
> different model, or an exception. That is why `sklearn_version` is in the
> metadata and why session 8's Dockerfile pins the environment. If you remember one
> deployment rule: **the model file and its environment travel together.**

---

## 2. The inference contract

Before writing the service, write down what it promises. This is the API
equivalent of session 1's specification, and it is what the team integrating with
you actually needs.

| | |
|---|---|
| **Endpoint** | `POST /predict` |
| **Input** | One application: seven numeric fields, three categorical. `annual_income` and `credit_score` may be null |
| **Output** | `probability_of_default` (0–1), `decision` (`approve` / `review`), `threshold`, `model_version` |
| **Errors** | `422` if a field is missing, the wrong type, or out of range |
| **Latency budget** | Under 5 seconds (session 1's charter); we should be far under |
| **Not promised** | An explanation, a batch mode, or a guarantee across model versions |

Two decisions worth noticing.

**We return the probability *and* the decision.** The probability lets the
consumer do something smarter later; the decision means they do not have to know
what a threshold is. Returning only one of them is a common design mistake.

**We return the threshold and the model version.** When somebody asks in March why
an application was flagged in January, those two fields are the answer.


```python
api_code = '''"""Loan default prediction service.

Run from the repository root:
    uvicorn app.api.main:app --reload

Then open http://127.0.0.1:8000/docs for the generated documentation.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Loaded ONCE, at startup — never inside the request handler.
model = joblib.load(MODEL_DIR / "loan_default_pipeline.joblib")
metadata = json.loads((MODEL_DIR / "loan_default_pipeline.json").read_text())

app = FastAPI(title="Loan default prediction", version=metadata["version"])


class Application(BaseModel):
    """One loan application, as it exists at the moment of submission."""

    loan_amount: float = Field(gt=0, le=50_000_000, examples=[800_000])
    tenure_months: int = Field(ge=1, le=120, examples=[24])
    interest_rate: float = Field(ge=0, le=50, examples=[14.5])
    previous_loans: int = Field(ge=0, le=50, examples=[0])
    days_past_due_history: int = Field(ge=0, le=3650, examples=[30])
    annual_income: Optional[float] = Field(default=None, ge=0)
    credit_score: Optional[float] = Field(default=None, ge=300, le=850)
    sector: str
    employment_type: str
    has_collateral: Literal["Yes", "No"]


class Prediction(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "review"]
    threshold: float
    model_version: str


@app.get("/health")
def health():
    """Liveness check — what a load balancer calls every few seconds."""
    return {"status": "ok", "model_version": metadata["version"]}


@app.get("/metadata")
def model_metadata():
    """What is deployed right now. The first thing to check in an incident."""
    return metadata


@app.post("/predict", response_model=Prediction)
def predict(application: Application) -> Prediction:
    frame = pd.DataFrame([application.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])

    return Prediction(
        probability_of_default=round(probability, 4),
        decision="review" if probability >= metadata["threshold"] else "approve",
        threshold=metadata["threshold"],
        model_version=metadata["version"],
    )
'''

API_PATH = ROOT / "app" / "api" / "main.py"
API_PATH.parent.mkdir(parents=True, exist_ok=True)
API_PATH.write_text(api_code)
print(f"wrote {API_PATH.relative_to(ROOT)}  ({len(api_code.splitlines())} lines)")
```

    wrote app/api/main.py  (71 lines)


Read that file rather than skimming it. Four details carry the lesson.

**The model is loaded once, at import.** Loading it inside `predict` would reread
the file on every request and turn a 2 millisecond answer into a 200 millisecond
one.

**`Application` is the schema, and the validation.** Pydantic enforces types and
ranges before your code sees anything. `loan_amount` must be positive,
`credit_score` must lie between 300 and 850, `has_collateral` must be exactly
`"Yes"` or `"No"`. A bad request is rejected with a readable error rather than
scored.

**Optional fields are explicit.** `annual_income` and `credit_score` may be
`None`, because sessions 2 and 3 established that missing values are normal here
and the pipeline's imputer handles them. Fields that may *not* be missing are
required, and that distinction is now enforced by the service.

**`/metadata` exists.** When a prediction is questioned, the first thing anyone
needs is what was deployed at the time.

---

## 3. Running it, and calling it

In a terminal you would start the service with:

```bash
uvicorn app.api.main:app --reload
```

Inside this notebook we start the same application on a background thread so we
can call it and shut it down cleanly.


```python
import threading

import requests
import uvicorn

# Import the app we just wrote, as a module, exactly as uvicorn would.
import sys
sys.path.insert(0, str(ROOT))
from app.api.main import app        # noqa: E402

PORT = 8123
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()

for _ in range(100):                # wait for startup, at most 10 seconds
    if server.started:
        break
    time.sleep(0.1)

BASE = f"http://127.0.0.1:{PORT}"
print("server started:", server.started)
print("health:", requests.get(f"{BASE}/health", timeout=5).json())
```

    server started: True
    health: {'status': 'ok', 'model_version': '1.0.0'}



```python
application = {
    "loan_amount": 800_000,
    "tenure_months": 24,
    "interest_rate": 14.5,
    "previous_loans": 0,
    "days_past_due_history": 30,
    "annual_income": 600_000,
    "credit_score": 545,
    "sector": "Construction",
    "employment_type": "Self-Employed",
    "has_collateral": "No",
}

response = requests.post(f"{BASE}/predict", json=application, timeout=5)
print(response.status_code)
print(json.dumps(response.json(), indent=2))
```

    200
    {
      "probability_of_default": 0.5311,
      "decision": "review",
      "threshold": 0.2,
      "model_version": "1.0.0"
    }


A real HTTP request, a real model, a real decision. That JSON is what the loan
origination system would receive.

Now the more interesting half: what happens when the request is wrong.


```python
broken = dict(application, loan_amount=-5000, credit_score=1200)

response = requests.post(f"{BASE}/predict", json=broken, timeout=5)
print("status:", response.status_code)
for error in response.json()["detail"]:
    print(f"  {error['loc'][-1]}: {error['msg']}")

missing_field = {k: v for k, v in application.items() if k != "sector"}
response = requests.post(f"{BASE}/predict", json=missing_field, timeout=5)
print("\nmissing a required field ->", response.status_code,
      response.json()["detail"][0]["msg"])
```

    status: 422
      loan_amount: Input should be greater than 0
      credit_score: Input should be less than or equal to 850
    
    missing a required field -> 422 Field required


**`422 Unprocessable Entity`**, with a message naming the offending field.

Compare that with what happens without validation: a negative loan amount is
perfectly acceptable arithmetic, the pipeline scales it, the model returns a
confident probability, and a nonsense decision travels downstream with no trace of
anything having gone wrong. Session 4's `reindex` warning was the notebook version
of this problem; Pydantic is the production answer.

The rule: **a service should refuse what it cannot score, loudly and early.**

---

## 4. Latency

Session 1's charter gave a budget of five seconds. Measure against it rather than
assuming.


```python
timings = []
for _ in range(50):
    start = time.perf_counter()
    requests.post(f"{BASE}/predict", json=application, timeout=5)
    timings.append((time.perf_counter() - start) * 1000)

timings = np.array(timings)
print(f"median  {np.median(timings):.1f} ms")
print(f"p95     {np.percentile(timings, 95):.1f} ms")
print(f"slowest {timings.max():.1f} ms")
print(f"\nbudget  5000 ms  —  headroom: {5000 / np.median(timings):.0f}x")
```

    median  2.3 ms
    p95     3.4 ms
    slowest 7.3 ms
    
    budget  5000 ms  —  headroom: 2202x


A couple of milliseconds per request, against a five-second budget. Three
observations that generalise beyond this project.

**Most of that time is HTTP, not the model.** The pipeline itself takes tens of
microseconds; the rest is request parsing, validation and serialisation. Which is
why "make the model faster" is usually the wrong first optimisation.

**Report the p95, not the mean.** Users experience the slow requests. A service
with a 2 ms median and a 4 second p95 is a broken service with a flattering
average.

**Latency is a design constraint, discovered early.** If the charter had demanded
predictions in 5 milliseconds, that would have ruled out the KNN model in session
4 before anyone trained it.


```python
# Shut the background server down so the notebook exits cleanly.
server.should_exit = True
thread.join(timeout=5)
print("server stopped:", not thread.is_alive())
```

    server stopped: True


---

## 5. An interface for humans

An API serves systems. A credit officer needs a screen. **Streamlit** turns a
Python script into a web interface without any front-end work, which makes it
ideal for internal tools and demonstrations.

The interface calls the API rather than loading the model itself. That separation
matters: one model, one place where the decision is made, and the UI can be
redeployed without touching the service.


```python
ui_code = '''"""Streamlit interface for the loan default service.

Start the API first:
    uvicorn app.api.main:app

Then, in a second terminal:
    streamlit run app/ui/app.py
"""

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("Loan default risk")
st.caption("Scores an application at submission time. Decision support, not a decision.")

with st.form("application"):
    column_left, column_right = st.columns(2)

    with column_left:
        loan_amount = st.number_input("Loan amount (NPR)", min_value=50_000, value=800_000, step=50_000)
        tenure_months = st.selectbox("Tenure (months)", [6, 12, 18, 24, 36, 48, 60], index=3)
        interest_rate = st.slider("Interest rate (%)", 8.0, 24.0, 14.5, 0.1)
        previous_loans = st.number_input("Previous loans", min_value=0, max_value=20, value=0)
        days_past_due_history = st.number_input("Days past due (history)", min_value=0, value=30)

    with column_right:
        sector = st.selectbox("Sector", ["Trade", "Service", "Manufacturing",
                                         "Agriculture", "Construction", "Tourism"])
        employment_type = st.selectbox("Employment", ["Salaried", "Self-Employed",
                                                      "Business Owner", "Informal"])
        has_collateral = st.radio("Collateral", ["Yes", "No"], horizontal=True)
        has_income = st.checkbox("Income documented", value=True)
        annual_income = st.number_input("Annual income (NPR)", min_value=0, value=600_000,
                                        step=50_000, disabled=not has_income)
        has_score = st.checkbox("Credit score available", value=True)
        credit_score = st.number_input("Credit score", min_value=300, max_value=850,
                                       value=545, disabled=not has_score)

    submitted = st.form_submit_button("Score application")

if submitted:
    payload = {
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "interest_rate": interest_rate,
        "previous_loans": previous_loans,
        "days_past_due_history": days_past_due_history,
        "annual_income": annual_income if has_income else None,
        "credit_score": credit_score if has_score else None,
        "sector": sector,
        "employment_type": employment_type,
        "has_collateral": has_collateral,
    }

    try:
        response = requests.post(f"{API}/predict", json=payload, timeout=5)
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API}. Is uvicorn running?")
        st.stop()

    if response.status_code != 200:
        st.error(f"The service rejected this application: {response.json()}")
        st.stop()

    result = response.json()
    probability = result["probability_of_default"]

    st.metric("Probability of default", f"{probability:.1%}")
    st.progress(min(probability / 0.5, 1.0))

    if result["decision"] == "review":
        st.warning("Send to senior review")
    else:
        st.success("Within auto-approval range")

    st.caption(f"threshold {result['threshold']:.0%} · model {result['model_version']}")
'''

UI_PATH = ROOT / "app" / "ui" / "app.py"
UI_PATH.parent.mkdir(parents=True, exist_ok=True)
UI_PATH.write_text(ui_code)
print(f"wrote {UI_PATH.relative_to(ROOT)}  ({len(ui_code.splitlines())} lines)")
print("\nTo use it, in two terminals from the repository root:")
print("  1)  uvicorn app.api.main:app")
print("  2)  streamlit run app/ui/app.py")
```

    wrote app/ui/app.py  (78 lines)
    
    To use it, in two terminals from the repository root:
      1)  uvicorn app.api.main:app
      2)  streamlit run app/ui/app.py


Notice three deliberate choices in that interface.

**Missing values are a checkbox, not a blank box.** "Income documented" reflects
the reality sessions 2 and 3 uncovered — an applicant with no documented income is
a normal case, not a form error. The UI sends `null` and the pipeline imputes.

**The probability and the recommendation are both shown.** The officer sees the
number and the suggested action, and remains the decision-maker. The caption
naming the threshold and model version is what makes a decision auditable months
later.

**Failures are handled.** If the API is down the interface says so. Demos that
crash on a disconnected backend lose more credibility than models that score 0.05
lower.

---

## 6. What a real service also needs

Our hundred lines are honest but incomplete. Before this served real applications,
the following would not be optional.

| Concern | What to add | Why |
|---------|-------------|-----|
| **Prediction logging** | Write every request and response to a store | Session 8's monitoring has nothing to measure otherwise |
| **Authentication** | An API key or service token | A scoring endpoint open to the internet is a data leak |
| **Rate limiting** | Requests per client per minute | One broken loop should not take the service down |
| **Timeouts and retries** | On both sides of the call | Networks fail; hanging requests are worse than failed ones |
| **Structured logging** | Request id on every line | Tracing one application through the system |
| **Containerisation** | Session 8's Dockerfile | The environment stops drifting |
| **Batch endpoint** | `POST /predict/batch` | Overnight scoring of the whole book |

The first row is the one that connects this session to the last one. **A
prediction that is not logged cannot be monitored**, and a service that is not
monitored will drift exactly as session 8's did — silently, for months.

The minimum log record: timestamp, request id, model version, the input fields,
the probability, the decision, and the response time. With those, every chart in
session 8 can be rebuilt from production traffic instead of from a CSV.

---

## Your turn

**1. Add prediction logging.** Append every request and response to a JSON Lines
file in `app/logs/`. Then load that file with pandas and reproduce session 8's
predicted-rate chart from it.

**2. Add a batch endpoint.** `POST /predict/batch` taking a list of applications
and returning a list of predictions. Measure its latency for 1, 10 and 100
applications — how much of the cost is per request rather than per application?

**3. Serve the size-aware thresholds.** Session 6 saved NPR 8.6m using a lower
threshold for large loans. Move that rule into the service (it belongs in the
metadata file, not in the code) and return which band was applied.

**4. Break it on purpose.** Send a `sector` value the model has never seen, such
as `"Fintech"`. Does the service reject it, or score it? Should the schema use a
`Literal` of known sectors, and what breaks when the lender opens a new sector
next year?

**5. Version endpoint discipline.** Retrain the model, bump `version` to `1.1.0`,
restart the API, and confirm `/metadata` reports the change. Write the two
sentences you would send the integrating team about the new version.

**6. Health check that means something.** `/health` currently returns `ok`
unconditionally. Make it score a fixed known application and check the result is
within tolerance, so a corrupted model file fails the check instead of passing it.

---

## If you remember nothing else

**Save the pipeline, not the model.** Preprocessing and estimator in one file is
what makes training and serving the same computation.

**A model file is not deployable on its own.** Threshold, schema, versions and
training data hash travel with it, or nobody can explain a decision six months
later.

**Write the inference contract before the code.** Inputs, outputs, errors,
latency, and what you are *not* promising.

**Validate at the boundary.** A service that scores a negative loan amount has
produced a confident, wrong, untraceable decision. `422` is a feature.

**Load the model once, at startup.** Not per request.

**Measure latency, report the p95.** We had a 5-second budget and a 2-millisecond
median; most of that was HTTP, not the model.

**Log every prediction.** Monitoring, incident analysis, and next year's retrain
all depend on the log you either wrote or did not.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Serialisation** | Writing a fitted object to a file — `joblib`, pickle, skops |
| **Model artifact** | The saved model file, plus the metadata it needs to be used |
| **Inference contract** | The documented promise a prediction service makes |
| **Endpoint** | One callable URL on a service, e.g. `POST /predict` |
| **Pydantic model** | A typed schema that validates request bodies |
| **422** | HTTP status for a well-formed request whose contents fail validation |
| **Health check** | An endpoint a load balancer polls to see if the service is alive |
| **p95 latency** | The response time 95% of requests come in under |
| **Cold start** | The first request after startup, before caches are warm |
| **Prediction log** | The record of every input and output, the raw material of monitoring |

## Further reading

- FastAPI documentation, *Tutorial — User Guide*, first ten pages; then
  *Response Model* and *Handling Errors*.
- Pydantic documentation, *Fields* and *Validators* — everything the schema in
  section 2 could enforce.
- Streamlit documentation, *Get started* and *API reference — Input widgets*.
- scikit-learn User Guide, *Model persistence* — the security and version
  caveats around pickle and joblib, and where `skops` fits.

---

**Next session:** *End-to-End ML Project* — the whole module in one pass on a
fresh problem, plus the project brief and grading rubric for your own submission.
