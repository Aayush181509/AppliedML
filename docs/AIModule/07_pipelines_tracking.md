# ML Pipelines and Experiment Tracking

**Applied ML in Production · Session 7**

---

Six sessions in, our modelling code is a sequence of instructions in a human
head: encode the categoricals, split, take the training medians, fill both sides,
fit the scaler on train, transform both, train, threshold at 0.20 — unless the
loan is large, in which case 0.15.

It works because we have been careful. It will stop working the first time
somebody runs the cells out of order, or trains on a fresh extract, or tries to
serve the model from a web service that has no idea what our training medians
were.

This session replaces that discipline with two pieces of machinery:

- a **`Pipeline`**, so the entire chain from raw columns to prediction is one
  object that cannot be assembled wrongly, and
- an **experiment tracker**, so every run's parameters, metrics and model file are
  recorded instead of remembered.

Neither improves the model. Both are what make a model something a team can own.

## How to work through this

Type the code. The pipeline section is the most valuable twenty lines in the
module — you will reuse them on every project you ever do.

Run each cell, read the output, then read the commentary. If a cell errors, run
from the top.

## Learning objectives

After this session you will be able to:

- Build a `ColumnTransformer` that treats numeric and categorical columns
  differently, inside a single `Pipeline`.
- Explain why a pipeline makes the session 2 leakage rule automatic.
- Cross-validate and grid-search a whole pipeline, preprocessing included.
- Add engineered features inside a pipeline rather than before it.
- List what has to be recorded for a run to be reproducible.
- Log parameters, metrics and models to **MLflow**, and compare runs.
- Reload a logged model and make a prediction from it.

## Setup

Note what is *not* here: no `get_dummies`, no `fillna`, no `StandardScaler`
applied by hand. The pipeline will do all of it.


```python
%matplotlib inline

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 130)


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

X = loans[NUMERIC + CATEGORICAL]        # raw columns, gaps and text included
y = loans["defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"train {X_train.shape}, test {X_test.shape}")
print(f"missing values still in the training data: {int(X_train.isna().sum().sum()):,}")
```

    train (9000, 10), test (3000, 10)
    missing values still in the training data: 1,499


---

## 1. What we have been doing by hand

Sessions 2 to 6 built this chain, in this order, every time:

1. Encode categorical columns to numbers.
2. Split into train and test.
3. Compute medians **on the training set**.
4. Fill gaps on both sides with those medians.
5. Fit a scaler **on the training set**; transform both.
6. Fit the model.
7. Apply a threshold.

Every step has a rule attached, and four separate things can go wrong: wrong
order, statistics computed on the wrong rows, a column list that drifts out of
sync, and a serving environment that has none of the intermediate objects.

A `Pipeline` turns all of it into one estimator with one `fit`.


```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", drop="first")),
])

preprocessing = ColumnTransformer([
    ("numeric", numeric_steps, NUMERIC),
    ("categorical", categorical_steps, CATEGORICAL),
])

model = Pipeline([
    ("prepare", preprocessing),
    ("classify", LogisticRegression(max_iter=1000)),
])

model
```




<style>.sk-global {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

.sk-global.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

.sk-global.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

.sk-global {
  color: var(--sklearn-color-text);
}

.sk-global pre {
  padding: 0;
}

.sk-global input.sk-hidden--visually {
  border: 0;
  clip-path: inset(100%);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

.sk-global div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

.sk-global div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

.sk-global div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

.sk-global div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

.sk-global div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

.sk-global div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

.sk-global div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

.sk-global div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

.sk-global div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

.sk-global div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

.sk-global div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
.sk-global label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

.sk-global label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

.sk-global label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

.sk-global label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

.sk-global div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

.sk-global input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

.sk-global div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
.sk-global div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

.sk-global div.sk-label label.sk-toggleable__label,
.sk-global div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
.sk-global div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
.sk-global div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

.sk-global div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

.sk-global div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
.sk-global div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
.sk-global div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

.sk-global a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

.sk-global a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
.sk-global a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

.sk-global a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.sk-top-container.sk-global {
  /* pydata-sphinx-theme hides overflow, so scrolling is disabled.
   We need to set it to !important and add tabindex="0" in the HTML
   to allow keyboard-only users to navigate the display. */
  overflow-x: scroll !important;
  max-width: 100%;
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover td {
    background-color: #e0e0e0;
}

.estimator-table table :is(td, th) {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td, .estimator-table th {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

td.fitted-att-type {
    white-space: preserve nowrap;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

@supports(anchor-name: --doc-link) {
    a.param-doc-link,
    a.param-doc-link:link,
    a.param-doc-link:visited {
    anchor-name: --doc-link;
    }
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

@supports(position-area: center right) {
    .param-doc-description {
    position-area: center right;
    position: fixed;
    margin-left: 0;
    }
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}

.features {
  font-family: monospace;
  cursor: pointer;
  background-color: var(--sklearn-color-unfitted-level-0);
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: .20em;
  margin-bottom: 0.5em;
  font-size: inherit; /* Needed for jupyter */
}

.features.fitted {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features summary {
  cursor: pointer;
  display: flex;
  margin-bottom: 0;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
  padding: .25em;
}

.features details[open] > summary {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
  border-radius: .20em 0 0 0;
}

.features.fitted details[open] > summary {
  background-color: var(--sklearn-color-fitted-level-2);
  border-radius: .20em 0 0 0;
}

.features details > summary .arrow::before {
  content: "▸";
  color: grey;
}

.features details[open] > summary .arrow::before {
  content: "▾";
}

.features details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-unfitted-level-2);
}

.features.fitted details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-fitted-level-2);
}

.features .features-container {
  max-width: 15em;
  max-height: 10em;
  overflow: auto;
  scrollbar-width: thin;
  padding: .25em 0.1rem;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 0 0 .5em .5em;
}

.features.fitted .features-container {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features .image-container {
  block-size: 1em;
  inline-size: 1em;
  padding: 0;
  margin: 0%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.features .copy-paste-icon {
  background-size: 1em 1em;
  width: 1em;
  height: 1em;
  filter: grayscale(100%) opacity(60%);
}

.features .features-container table {
  width: 100%;
  margin: 0.01em;
}

.features .features-container table tr:nth-child(odd) {
  background-color: #fff;
}

.features .features-container table tr:nth-child(even) {
  background-color: #f6f6f6;
}

.features .features-container table tr:hover {
  background-color: #e0e0e0;
}

.features .features-container table {
  table-layout: inherit;
}

.features .features-container table td {
  text-align: left;
  padding: 0 0.5em;
  border: 1px solid rgba(106, 105, 104, 0.232);
  white-space: nowrap;
  color: var(--sklearn-color-text);
}

.total_features {
  display: flex;
  justify-content: center;
  margin-top: 0.5em;
}
</style><body><div id="sk-container-id-1" tabindex="0" class="sk-top-container sk-global"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;prepare&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;impute&#x27;,
                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),
                                                                  (&#x27;scale&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;loan_amount&#x27;,
                                                   &#x27;tenure_months&#x27;,
                                                   &#x27;interest_rate&#x27;,
                                                   &#x27;previous_loans&#x27;,
                                                   &#x27;days_past_due_history&#x27;,
                                                   &#x27;annual_income&#x27;,
                                                   &#x27;credit_score&#x27;]),
                                                 (&#x27;categorical&#x27;,
                                                  Pipeline(steps=[(&#x27;impute&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;encode&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;))]),
                                                  [&#x27;sector&#x27;, &#x27;employment_type&#x27;,
                                                   &#x27;has_collateral&#x27;])])),
                (&#x27;classify&#x27;, LogisticRegression(max_iter=1000))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></div></label><div class="sk-toggleable__content " data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('steps',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-steps;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=steps,-list%20of%20tuples">
            steps
            <span class="param-doc-description"
            style="position-anchor: --doc-link-steps;">
            steps: list of tuples<br><br>List of (name of step, estimator) tuples that are to be chained in<br>sequential order. To be compatible with the scikit-learn API, all steps<br>must define `fit`. All non-last steps must also define `transform`. See<br>:ref:`Combining Estimators &lt;combining_estimators&gt;` for more details.</span>
        </a>
    </td>
            <td class="value">[(&#x27;prepare&#x27;, ...), (&#x27;classify&#x27;, ...)]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transform_input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-transform_input;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=transform_input,-list%20of%20str%2C%20default%3DNone">
            transform_input
            <span class="param-doc-description"
            style="position-anchor: --doc-link-transform_input;">
            transform_input: list of str, default=None<br><br>The names of the :term:`metadata` parameters that should be transformed by the<br>pipeline before passing it to the step consuming it.<br><br>This enables transforming some input arguments to ``fit`` (other than ``X``)<br>to be transformed by the steps of the pipeline up to the step which requires<br>them. Requirement is defined via :ref:`metadata routing &lt;metadata_routing&gt;`.<br>For instance, this can be used to pass a validation set through the pipeline.<br><br>You can only set this if metadata routing is enabled, which you<br>can enable using ``sklearn.set_config(enable_metadata_routing=True)``.<br><br>.. versionadded:: 1.6</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('memory',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-memory;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=memory,-str%20or%20object%20with%20the%20joblib.Memory%20interface%2C%20default%3DNone">
            memory
            <span class="param-doc-description"
            style="position-anchor: --doc-link-memory;">
            memory: str or object with the joblib.Memory interface, default=None<br><br>Used to cache the fitted transformers of the pipeline. The last step<br>will never be cached, even if it is a transformer. By default, no<br>caching is performed. If a string is given, it is the path to the<br>caching directory. Enabling caching triggers a clone of the transformers<br>before fitting. Therefore, the transformer instance given to the<br>pipeline cannot be inspected directly. Use the attribute ``named_steps``<br>or ``steps`` to inspect estimators within the pipeline. Caching the<br>transformers is advantageous when fitting is time consuming. See<br>:ref:`sphx_glr_auto_examples_neighbors_plot_caching_nearest_neighbors.py`<br>for an example on how to enable caching.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=verbose,-bool%2C%20default%3DFalse">
            verbose
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose;">
            verbose: bool, default=False<br><br>If True, the time elapsed while fitting each step will be printed as it<br>is completed.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>prepare: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for prepare: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="prepare__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformers',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-transformers;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=transformers,-list%20of%20tuples">
            transformers
            <span class="param-doc-description"
            style="position-anchor: --doc-link-transformers;">
            transformers: list of tuples<br><br>List of (name, transformer, columns) tuples specifying the<br>transformer objects to be applied to subsets of the data.<br><br>name : str<br>    Like in Pipeline and FeatureUnion, this allows the transformer and<br>    its parameters to be set using ``set_params`` and searched in grid<br>    search.<br>transformer : {&#x27;drop&#x27;, &#x27;passthrough&#x27;} or estimator<br>    Estimator must support :term:`fit` and :term:`transform`.<br>    Special-cased strings &#x27;drop&#x27; and &#x27;passthrough&#x27; are accepted as<br>    well, to indicate to drop the columns or to pass them through<br>    untransformed, respectively.<br>columns :  str, array-like of str, int, array-like of int,                 array-like of bool, slice or callable<br>    Indexes the data on its second axis. Integers are interpreted as<br>    positional columns, while strings can reference DataFrame columns<br>    by name.  A scalar string or int should be used where<br>    ``transformer`` expects X to be a 1d array-like (vector),<br>    otherwise a 2d array will be passed to the transformer.<br>    A callable is passed the input data `X` and can return any of the<br>    above. To select multiple columns by name or dtype, you can use<br>    :obj:`make_column_selector`.</span>
        </a>
    </td>
            <td class="value">[(&#x27;numeric&#x27;, ...), (&#x27;categorical&#x27;, ...)]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('remainder',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-remainder;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=remainder,-%7B%27drop%27%2C%20%27passthrough%27%7D%20or%20estimator%2C%20default%3D%27drop%27">
            remainder
            <span class="param-doc-description"
            style="position-anchor: --doc-link-remainder;">
            remainder: {&#x27;drop&#x27;, &#x27;passthrough&#x27;} or estimator, default=&#x27;drop&#x27;<br><br>By default, only the specified columns in `transformers` are<br>transformed and combined in the output, and the non-specified<br>columns are dropped. (default of ``&#x27;drop&#x27;``).<br>By specifying ``remainder=&#x27;passthrough&#x27;``, all remaining columns that<br>were not specified in `transformers`, but present in the data passed<br>to `fit` will be automatically passed through. This subset of columns<br>is concatenated with the output of the transformers. For dataframes,<br>extra columns not seen during `fit` will be excluded from the output<br>of `transform`.<br>By setting ``remainder`` to be an estimator, the remaining<br>non-specified columns will use the ``remainder`` estimator. The<br>estimator must support :term:`fit` and :term:`transform`.<br>Note that using this feature requires that the DataFrame columns<br>input at :term:`fit` and :term:`transform` have identical order.</span>
        </a>
    </td>
            <td class="value">&#x27;drop&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sparse_threshold',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-sparse_threshold;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=sparse_threshold,-float%2C%20default%3D0.3">
            sparse_threshold
            <span class="param-doc-description"
            style="position-anchor: --doc-link-sparse_threshold;">
            sparse_threshold: float, default=0.3<br><br>If the output of the different transformers contains sparse matrices,<br>these will be stacked as a sparse matrix if the overall density is<br>lower than this value. Use ``sparse_threshold=0`` to always return<br>dense.  When the transformed output consists of all dense data, the<br>stacked result will be dense, and this keyword will be ignored.</span>
        </a>
    </td>
            <td class="value">0.3</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_jobs;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=n_jobs,-int%2C%20default%3DNone">
            n_jobs
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_jobs;">
            n_jobs: int, default=None<br><br>Number of jobs to run in parallel.<br>``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.<br>``-1`` means using all processors. See :term:`Glossary &lt;n_jobs&gt;`<br>for more details.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformer_weights',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-transformer_weights;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=transformer_weights,-dict%2C%20default%3DNone">
            transformer_weights
            <span class="param-doc-description"
            style="position-anchor: --doc-link-transformer_weights;">
            transformer_weights: dict, default=None<br><br>Multiplicative weights for features per transformer. The output of the<br>transformer is multiplied by these weights. Keys are transformer names,<br>values the weights.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=verbose,-bool%2C%20default%3DFalse">
            verbose
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose;">
            verbose: bool, default=False<br><br>If True, the time elapsed while fitting each transformer will be<br>printed as it is completed.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose_feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose_feature_names_out;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=verbose_feature_names_out,-bool%2C%20str%20or%20Callable%5B%5Bstr%2C%20str%5D%2C%20str%5D%2C%20default%3DTrue">
            verbose_feature_names_out
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose_feature_names_out;">
            verbose_feature_names_out: bool, str or Callable[[str, str], str], default=True<br><br>- If True, :meth:`ColumnTransformer.get_feature_names_out` will prefix<br>  all feature names with the name of the transformer that generated that<br>  feature. It is equivalent to setting<br>  `verbose_feature_names_out=&quot;{transformer_name}__{feature_name}&quot;`.<br>- If False, :meth:`ColumnTransformer.get_feature_names_out` will not<br>  prefix any feature names and will error if feature names are not<br>  unique.<br>- If ``Callable[[str, str], str]``,<br>  :meth:`ColumnTransformer.get_feature_names_out` will rename all the features<br>  using the name of the transformer. The first argument of the callable is the<br>  transformer name and the second argument is the feature name. The returned<br>  string will be the new feature name.<br>- If ``str``, it must be a string ready for formatting. The given string will<br>  be formatted using two field names: ``transformer_name`` and ``feature_name``.<br>  e.g. ``&quot;{feature_name}__{transformer_name}&quot;``. See :meth:`str.format` method<br>  from the standard library for more info.<br><br>.. versionadded:: 1.0<br><br>.. versionchanged:: 1.6<br>    `verbose_feature_names_out` can be a callable or a string to be formatted.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>numeric</div></div></label><div class="sk-toggleable__content " data-param-prefix="prepare__numeric__"><pre>[&#x27;loan_amount&#x27;, &#x27;tenure_months&#x27;, &#x27;interest_rate&#x27;, &#x27;previous_loans&#x27;, &#x27;days_past_due_history&#x27;, &#x27;annual_income&#x27;, &#x27;credit_score&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="prepare__numeric__impute__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-strategy;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description"
            style="position-anchor: --doc-link-strategy;">
            strategy: str or Callable, default=&#x27;mean&#x27;<br><br>The imputation strategy.<br><br>- If &quot;mean&quot;, then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If &quot;median&quot;, then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If &quot;most_frequent&quot;, then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If &quot;constant&quot;, then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy=&quot;constant&quot; for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;median&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-missing_values;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description"
            style="position-anchor: --doc-link-missing_values;">
            missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas&#x27; dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-fill_value;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description"
            style="position-anchor: --doc-link-fill_value;">
            fill_value: str or numerical value, default=None<br><br>When strategy == &quot;constant&quot;, `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and &quot;missing_value&quot; for strings or object data types.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-copy;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description"
            style="position-anchor: --doc-link-copy;">
            copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-add_indicator;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description"
            style="position-anchor: --doc-link-add_indicator;">
            add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer&#x27;s transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won&#x27;t appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-keep_empty_features;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description"
            style="position-anchor: --doc-link-keep_empty_features;">
            keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy=&quot;constant&quot;`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="prepare__numeric__scale__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-copy;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description"
            style="position-anchor: --doc-link-copy;">
            copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-with_mean;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description"
            style="position-anchor: --doc-link-with_mean;">
            with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-with_std;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description"
            style="position-anchor: --doc-link-with_std;">
            with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>categorical</div></div></label><div class="sk-toggleable__content " data-param-prefix="prepare__categorical__"><pre>[&#x27;sector&#x27;, &#x27;employment_type&#x27;, &#x27;has_collateral&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="prepare__categorical__impute__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-strategy;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description"
            style="position-anchor: --doc-link-strategy;">
            strategy: str or Callable, default=&#x27;mean&#x27;<br><br>The imputation strategy.<br><br>- If &quot;mean&quot;, then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If &quot;median&quot;, then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If &quot;most_frequent&quot;, then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If &quot;constant&quot;, then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy=&quot;constant&quot; for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;most_frequent&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-missing_values;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description"
            style="position-anchor: --doc-link-missing_values;">
            missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas&#x27; dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-fill_value;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description"
            style="position-anchor: --doc-link-fill_value;">
            fill_value: str or numerical value, default=None<br><br>When strategy == &quot;constant&quot;, `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and &quot;missing_value&quot; for strings or object data types.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-copy;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description"
            style="position-anchor: --doc-link-copy;">
            copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-add_indicator;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description"
            style="position-anchor: --doc-link-add_indicator;">
            add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer&#x27;s transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won&#x27;t appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-keep_empty_features;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description"
            style="position-anchor: --doc-link-keep_empty_features;">
            keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy=&quot;constant&quot;`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="prepare__categorical__encode__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('drop',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-drop;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=drop,-%7B%27first%27%2C%20%27if_binary%27%7D%20or%20an%20array-like%20of%20shape%20%28n_features%2C%29%2C%20%20%20%20%20%20%20%20%20%20%20%20%20default%3DNone">
            drop
            <span class="param-doc-description"
            style="position-anchor: --doc-link-drop;">
            drop: {&#x27;first&#x27;, &#x27;if_binary&#x27;} or an array-like of shape (n_features,),             default=None<br><br>Specifies a methodology to use to drop one of the categories per<br>feature. This is useful in situations where perfectly collinear<br>features cause problems, such as when feeding the resulting data<br>into an unregularized linear regression model.<br><br>However, dropping one category breaks the symmetry of the original<br>representation and can therefore induce a bias in downstream models,<br>for instance for penalized linear classification or regression models.<br><br>- None : retain all features (the default).<br>- &#x27;first&#x27; : drop the first category in each feature. If only one<br>  category is present, the feature will be dropped entirely.<br>- &#x27;if_binary&#x27; : drop the first category in each feature with two<br>  categories. Features with 1 or more than 2 categories are<br>  left intact.<br>- array : ``drop[i]`` is the category in feature ``X[:, i]`` that<br>  should be dropped.<br><br>When `max_categories` or `min_frequency` is configured to group<br>infrequent categories, the dropping behavior is handled after the<br>grouping.<br><br>.. versionadded:: 0.21<br>   The parameter `drop` was added in 0.21.<br><br>.. versionchanged:: 0.23<br>   The option `drop=&#x27;if_binary&#x27;` was added in 0.23.<br><br>.. versionchanged:: 1.1<br>    Support for dropping infrequent categories.</span>
        </a>
    </td>
            <td class="value">&#x27;first&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-handle_unknown;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27ignore%27%2C%20%27infrequent_if_exist%27%2C%20%27warn%27%7D%2C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description"
            style="position-anchor: --doc-link-handle_unknown;">
            handle_unknown: {&#x27;error&#x27;, &#x27;ignore&#x27;, &#x27;infrequent_if_exist&#x27;, &#x27;warn&#x27;},                      default=&#x27;error&#x27;<br><br>Specifies the way unknown categories are handled during :meth:`transform`.<br><br>- &#x27;error&#x27; : Raise an error if an unknown category is present during transform.<br>- &#x27;ignore&#x27; : When an unknown category is encountered during<br>  transform, the resulting one-hot encoded columns for this feature<br>  will be all zeros. In the inverse transform, an unknown category<br>  will be denoted as None.<br>- &#x27;infrequent_if_exist&#x27; : When an unknown category is encountered<br>  during transform, the resulting one-hot encoded columns for this<br>  feature will map to the infrequent category if it exists. The<br>  infrequent category will be mapped to the last position in the<br>  encoding. During inverse transform, an unknown category will be<br>  mapped to the category denoted `&#x27;infrequent&#x27;` if it exists. If the<br>  `&#x27;infrequent&#x27;` category does not exist, then :meth:`transform` and<br>  :meth:`inverse_transform` will handle an unknown category as with<br>  `handle_unknown=&#x27;ignore&#x27;`. Infrequent categories exist based on<br>  `min_frequency` and `max_categories`. Read more in the<br>  :ref:`User Guide &lt;encoder_infrequent_categories&gt;`.<br>- &#x27;warn&#x27; : When an unknown category is encountered during transform<br>  a warning is issued, and the encoding then proceeds as described for<br>  `handle_unknown=&quot;infrequent_if_exist&quot;`.<br><br>.. versionchanged:: 1.1<br>    `&#x27;infrequent_if_exist&#x27;` was added to automatically handle unknown<br>    categories and infrequent categories.<br><br>.. versionadded:: 1.6<br>   The option `&quot;warn&quot;` was added in 1.6.</span>
        </a>
    </td>
            <td class="value">&#x27;ignore&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-categories;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description"
            style="position-anchor: --doc-link-categories;">
            categories: &#x27;auto&#x27; or a list of array-like, default=&#x27;auto&#x27;<br><br>Categories (unique values) per feature:<br><br>- &#x27;auto&#x27; : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values within a single feature, and should be sorted in case of<br>  numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">&#x27;auto&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sparse_output',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-sparse_output;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=sparse_output,-bool%2C%20default%3DTrue">
            sparse_output
            <span class="param-doc-description"
            style="position-anchor: --doc-link-sparse_output;">
            sparse_output: bool, default=True<br><br>When ``True``, it returns a SciPy sparse matrix/array<br>in &quot;Compressed Sparse Row&quot; (CSR) format.<br><br>.. versionadded:: 1.2<br>   `sparse` was renamed to `sparse_output`</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-dtype;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description"
            style="position-anchor: --doc-link-dtype;">
            dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_frequency;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_frequency;">
            min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.1<br>    Read more in the :ref:`User Guide &lt;encoder_infrequent_categories&gt;`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_categories;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_categories;">
            max_categories: int, default=None<br><br>Specifies an upper limit to the number of output features for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>.. versionadded:: 1.1<br>    Read more in the :ref:`User Guide &lt;encoder_infrequent_categories&gt;`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_name_combiner',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_name_combiner;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=feature_name_combiner,-%22concat%22%20or%20callable%2C%20default%3D%22concat%22">
            feature_name_combiner
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_name_combiner;">
            feature_name_combiner: &quot;concat&quot; or callable, default=&quot;concat&quot;<br><br>Callable with signature `def callable(input_feature, category)` that returns a<br>string. This is used to create feature names to be returned by<br>:meth:`get_feature_names_out`.<br><br>`&quot;concat&quot;` concatenates encoded feature name and category with<br>`feature + &quot;_&quot; + str(category)`.E.g. feature X with values 1, 6, 7 create<br>feature names `X_1, X_6, X_7`.<br><br>.. versionadded:: 1.3</span>
        </a>
    </td>
            <td class="value">&#x27;concat&#x27;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>LogisticRegression</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="classify__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_iter',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_iter;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=max_iter,-int%2C%20default%3D100">
            max_iter
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_iter;">
            max_iter: int, default=100<br><br>Maximum number of iterations taken for the solvers to converge.</span>
        </a>
    </td>
            <td class="value">1000</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('penalty',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-penalty;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=penalty,-%7B%27l1%27%2C%20%27l2%27%2C%20%27elasticnet%27%2C%20None%7D%2C%20default%3D%27l2%27">
            penalty
            <span class="param-doc-description"
            style="position-anchor: --doc-link-penalty;">
            penalty: {&#x27;l1&#x27;, &#x27;l2&#x27;, &#x27;elasticnet&#x27;, None}, default=&#x27;l2&#x27;<br><br>Specify the norm of the penalty:<br><br>- `None`: no penalty is added;<br>- `&#x27;l2&#x27;`: add an L2 penalty term and it is the default choice;<br>- `&#x27;l1&#x27;`: add an L1 penalty term;<br>- `&#x27;elasticnet&#x27;`: both L1 and L2 penalty terms are added.<br><br>.. warning::<br>   Some penalties may not work with some solvers. See the parameter<br>   `solver` below, to know the compatibility between the penalty and<br>   solver.<br><br>.. versionadded:: 0.19<br>   l1 penalty with SAGA solver (allowing &#x27;multinomial&#x27; + L1)<br><br>.. deprecated:: 1.8<br>   `penalty` was deprecated in version 1.8 and will be removed in 1.10.<br>   Use `l1_ratio` and `C` instead. `l1_ratio=0` for `penalty=&#x27;l2&#x27;`,<br>   `l1_ratio=1` for `penalty=&#x27;l1&#x27;`, `l1_ratio` set to any float between 0 and 1<br>   for `penalty=&#x27;elasticnet&#x27;`, and `C=np.inf` for `penalty=None`.</span>
        </a>
    </td>
            <td class="value">&#x27;deprecated&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('C',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-C;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=C,-float%2C%20default%3D1.0">
            C
            <span class="param-doc-description"
            style="position-anchor: --doc-link-C;">
            C: float, default=1.0<br><br>Inverse of regularization strength; must be a positive float.<br>Like in support vector machines, smaller values specify stronger<br>regularization. `C=np.inf` results in unpenalized logistic regression.<br>For a visual example on the effect of tuning the `C` parameter<br>with an L1 penalty, see:<br>:ref:`sphx_glr_auto_examples_linear_model_plot_logistic_path.py`.</span>
        </a>
    </td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('l1_ratio',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-l1_ratio;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=l1_ratio,-float%2C%20default%3D0.0">
            l1_ratio
            <span class="param-doc-description"
            style="position-anchor: --doc-link-l1_ratio;">
            l1_ratio: float, default=0.0<br><br>The Elastic-Net mixing parameter, with `0 &lt;= l1_ratio &lt;= 1`. Setting<br>`l1_ratio=1` gives a pure L1-penalty, setting `l1_ratio=0` a pure L2-penalty.<br>Any value between 0 and 1 gives an Elastic-Net penalty of the form<br>`l1_ratio * L1 + (1 - l1_ratio) * L2`.<br><br>.. warning::<br>   Certain values of `l1_ratio`, i.e. some penalties, may not work with some<br>   solvers. See the parameter `solver` below, to know the compatibility between<br>   the penalty and solver.<br><br>.. versionchanged:: 1.8<br>    Default value changed from None to 0.0.<br><br>.. deprecated:: 1.8<br>    `None` is deprecated and will be removed in version 1.10. Always use<br>    `l1_ratio` to specify the penalty type.</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dual',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-dual;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=dual,-bool%2C%20default%3DFalse">
            dual
            <span class="param-doc-description"
            style="position-anchor: --doc-link-dual;">
            dual: bool, default=False<br><br>Dual (constrained) or primal (regularized, see also<br>:ref:`this equation &lt;regularized-logistic-loss&gt;`) formulation. Dual formulation<br>is only implemented for l2 penalty with liblinear solver. Prefer `dual=False`<br>when n_samples &gt; n_features.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tol',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-tol;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=tol,-float%2C%20default%3D1e-4">
            tol
            <span class="param-doc-description"
            style="position-anchor: --doc-link-tol;">
            tol: float, default=1e-4<br><br>Tolerance for stopping criteria.</span>
        </a>
    </td>
            <td class="value">0.0001</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fit_intercept',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-fit_intercept;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=fit_intercept,-bool%2C%20default%3DTrue">
            fit_intercept
            <span class="param-doc-description"
            style="position-anchor: --doc-link-fit_intercept;">
            fit_intercept: bool, default=True<br><br>Specifies if a constant (a.k.a. bias or intercept) should be<br>added to the decision function.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('intercept_scaling',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-intercept_scaling;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=intercept_scaling,-float%2C%20default%3D1">
            intercept_scaling
            <span class="param-doc-description"
            style="position-anchor: --doc-link-intercept_scaling;">
            intercept_scaling: float, default=1<br><br>Useful only when the solver `liblinear` is used<br>and `self.fit_intercept` is set to `True`. In this case, `x` becomes<br>`[x, self.intercept_scaling]`,<br>i.e. a &quot;synthetic&quot; feature with constant value equal to<br>`intercept_scaling` is appended to the instance vector.<br>The intercept becomes<br>``intercept_scaling * synthetic_feature_weight``.<br><br>.. note::<br>    The synthetic feature weight is subject to L1 or L2<br>    regularization as all other features.<br>    To lessen the effect of regularization on synthetic feature weight<br>    (and therefore on the intercept) `intercept_scaling` has to be increased.</span>
        </a>
    </td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('class_weight',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-class_weight;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=class_weight,-dict%20or%20%27balanced%27%2C%20default%3DNone">
            class_weight
            <span class="param-doc-description"
            style="position-anchor: --doc-link-class_weight;">
            class_weight: dict or &#x27;balanced&#x27;, default=None<br><br>Weights associated with classes in the form ``{class_label: weight}``.<br>If not given, all classes are supposed to have weight one.<br><br>The &quot;balanced&quot; mode uses the values of y to automatically adjust<br>weights inversely proportional to class frequencies in the input data<br>as ``n_samples / (n_classes * np.bincount(y))``.<br><br>Note that these weights will be multiplied with sample_weight (passed<br>through the fit method) if sample_weight is specified.<br><br>.. versionadded:: 0.17<br>   *class_weight=&#x27;balanced&#x27;*</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-random_state;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=random_state,-int%2C%20RandomState%20instance%2C%20default%3DNone">
            random_state
            <span class="param-doc-description"
            style="position-anchor: --doc-link-random_state;">
            random_state: int, RandomState instance, default=None<br><br>Used when ``solver`` == &#x27;sag&#x27;, &#x27;saga&#x27; or &#x27;liblinear&#x27; to shuffle the<br>data. See :term:`Glossary &lt;random_state&gt;` for details.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('solver',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-solver;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=solver,-%7B%27lbfgs%27%2C%20%27liblinear%27%2C%20%27newton-cg%27%2C%20%27newton-cholesky%27%2C%20%27sag%27%2C%20%27saga%27%7D%2C%20%20%20%20%20%20%20%20%20%20%20%20%20default%3D%27lbfgs%27">
            solver
            <span class="param-doc-description"
            style="position-anchor: --doc-link-solver;">
            solver: {&#x27;lbfgs&#x27;, &#x27;liblinear&#x27;, &#x27;newton-cg&#x27;, &#x27;newton-cholesky&#x27;, &#x27;sag&#x27;, &#x27;saga&#x27;},             default=&#x27;lbfgs&#x27;<br><br>Algorithm to use in the optimization problem. Default is &#x27;lbfgs&#x27;.<br>To choose a solver, you might want to consider the following aspects:<br><br>- &#x27;lbfgs&#x27; is a good default solver because it works reasonably well for a wide<br>  class of problems.<br>- For :term:`multiclass` problems (`n_classes &gt;= 3`), all solvers except<br>  &#x27;liblinear&#x27; minimize the full multinomial loss, &#x27;liblinear&#x27; will raise an<br>  error.<br>- &#x27;newton-cholesky&#x27; is a good choice for<br>  `n_samples` &gt;&gt; `n_features * n_classes`, especially with one-hot encoded<br>  categorical features with rare categories. Be aware that the memory usage<br>  of this solver has a quadratic dependency on `n_features * n_classes`<br>  because it explicitly computes the full Hessian matrix.<br>- For small datasets, &#x27;liblinear&#x27; is a good choice, whereas &#x27;sag&#x27;<br>  and &#x27;saga&#x27; are faster for large ones;<br>- &#x27;liblinear&#x27; can only handle binary classification by default. To apply a<br>  one-versus-rest scheme for the multiclass setting one can wrap it with the<br>  :class:`~sklearn.multiclass.OneVsRestClassifier`.<br><br>.. warning::<br>   The choice of the algorithm depends on the penalty chosen (`l1_ratio=0`<br>   for L2-penalty, `l1_ratio=1` for L1-penalty and `0 &lt; l1_ratio &lt; 1` for<br>   Elastic-Net) and on (multinomial) multiclass support:<br><br>   ================= ======================== ======================<br>   solver            l1_ratio                 multinomial multiclass<br>   ================= ======================== ======================<br>   &#x27;lbfgs&#x27;           l1_ratio=0               yes<br>   &#x27;liblinear&#x27;       l1_ratio=1 or l1_ratio=0 no<br>   &#x27;newton-cg&#x27;       l1_ratio=0               yes<br>   &#x27;newton-cholesky&#x27; l1_ratio=0               yes<br>   &#x27;sag&#x27;             l1_ratio=0               yes<br>   &#x27;saga&#x27;            0&lt;=l1_ratio&lt;=1           yes<br>   ================= ======================== ======================<br><br>.. note::<br>   &#x27;sag&#x27; and &#x27;saga&#x27; fast convergence is only guaranteed on features<br>   with approximately the same scale. You can preprocess the data with<br>   a scaler from :mod:`sklearn.preprocessing`.<br><br>.. seealso::<br>   Refer to the :ref:`User Guide &lt;Logistic_regression&gt;` for more<br>   information regarding :class:`LogisticRegression` and more specifically the<br>   :ref:`Table &lt;logistic_regression_solvers&gt;`<br>   summarizing solver/penalty supports.<br><br>.. versionadded:: 0.17<br>   Stochastic Average Gradient (SAG) descent solver. Multinomial support in<br>   version 0.18.<br>.. versionadded:: 0.19<br>   SAGA solver.<br>.. versionchanged:: 0.22<br>   The default solver changed from &#x27;liblinear&#x27; to &#x27;lbfgs&#x27; in 0.22.<br>.. versionadded:: 1.2<br>   newton-cholesky solver. Multinomial support in version 1.6.</span>
        </a>
    </td>
            <td class="value">&#x27;lbfgs&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=verbose,-int%2C%20default%3D0">
            verbose
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose;">
            verbose: int, default=0<br><br>For the liblinear and lbfgs solvers set verbose to any positive<br>number for verbosity.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-warm_start;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=warm_start,-bool%2C%20default%3DFalse">
            warm_start
            <span class="param-doc-description"
            style="position-anchor: --doc-link-warm_start;">
            warm_start: bool, default=False<br><br>When set to True, reuse the solution of the previous call to fit as<br>initialization, otherwise, just erase the previous solution.<br>Useless for liblinear solver. See :term:`the Glossary &lt;warm_start&gt;`.<br><br>.. versionadded:: 0.17<br>   *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_jobs;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html#:~:text=n_jobs,-int%2C%20default%3DNone">
            n_jobs
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_jobs;">
            n_jobs: int, default=None<br><br>Does not have any effect.<br><br>.. deprecated:: 1.8<br>   `n_jobs` is deprecated in version 1.8 and will be removed in 1.10.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div><script>/*  Authors: The scikit-learn developers
 SPDX-License-Identifier: BSD-3-Clause
*/

function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';

    const parent = element.parentElement;
    if (!parent || !parent.nextElementSibling) {
        console.warn('Expected copy-paste icon is missing from the DOM structure');
        return;
    }

    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});

/**
 * Copy the list of feature names formatted as a Python list.
 *
 * @param {HTMLElement} element - The copy button inside a `.features` block; its siblings
 *   contain a `details` element and a table containing feature named.
 * @returns {boolean} Always returns `false` so callers can prevent the default click behavior.
 */
function copyFeatureNamesToClipboard(element) {
    var detailsElem = element.closest('.features').querySelector('details');
    var wasOpen = detailsElem.open;
    detailsElem.open = true;
    var content = element.closest('.features').querySelector('tbody')
                  .innerText.trim();
    if (!wasOpen) detailsElem.open = false;
    const rows = content.split('\n').map(row => `    "${row}"`);
    const formattedText = `[\n${rows.join(',\n')},\n]`;
    const originalHTML = element.innerHTML.replace('✔', '');
    const originalStyle = element.style;
    const copyMark = document.createElement('span');
    copyMark.innerHTML = '✔';
    copyMark.style.color = 'blue';
    copyMark.style.fontSize = '1em';

    navigator.clipboard.writeText(formattedText)
        .then(() => {
            element.style.display = 'none';
            element.parentElement.appendChild(copyMark);

            setTimeout(() => {
                copyMark.remove();
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'orange';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        });
    return false;
}
/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-1');</script></body>



Read the structure above. `ColumnTransformer` sends the numeric columns down one
branch and the categorical columns down another, then glues the results back
together. The whole thing is one estimator, so it still obeys session 4's
contract: `fit`, `predict`, `predict_proba`.


```python
from sklearn.metrics import average_precision_score, roc_auc_score

model.fit(X_train, y_train)                       # raw frame in, trained model out
probabilities = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC            {roc_auc_score(y_test, probabilities):.3f}")
print(f"average precision  {average_precision_score(y_test, probabilities):.3f}")
```

    ROC-AUC            0.782
    average precision  0.382


The same numbers as session 5, from one `fit` on raw columns with missing values
and text in them.

Four things came free with that:

**Leakage is now structural.** The imputer and scaler are fitted inside `fit`, on
whatever rows `fit` was given. There is no way to accidentally include the test
set — the code that could do it no longer exists.

**Unseen categories will not crash it.** `handle_unknown="ignore"` means a branch
or sector the model never saw becomes all zeros rather than an exception. Session
2 flagged this as the reason `get_dummies` cannot be used in production.

**The column list lives in one place.** `NUMERIC` and `CATEGORICAL` are declared
once and used by the pipeline forever.

**It is one object.** One thing to cross-validate, one thing to save, one thing to
load in a web service — which is session 9.


```python
# Inspect what the pipeline built: names come out of the fitted encoder.
encoded_names = model.named_steps["prepare"].get_feature_names_out()
print(f"{len(encoded_names)} columns after preprocessing")
print(list(encoded_names[:5]), "...")

coefficients = pd.Series(model.named_steps["classify"].coef_[0], index=encoded_names)
print("\nstrongest three:")
print(coefficients.abs().sort_values(ascending=False).head(3).round(3).to_string())
```

    16 columns after preprocessing
    ['numeric__loan_amount', 'numeric__tenure_months', 'numeric__interest_rate', 'numeric__previous_loans', 'numeric__days_past_due_history'] ...
    
    strongest three:
    numeric__interest_rate                   0.618
    categorical__employment_type_Informal    0.410
    numeric__annual_income                   0.331


---

## 2. Cross-validating the whole thing

This is where a pipeline stops being tidiness and starts being correctness.

When you cross-validate a pipeline, scikit-learn refits **every step** on each
training fold — including the imputer's medians and the scaler's statistics. Do
it by hand and those statistics are computed once, over all the training data,
including the rows being held out for validation. The result is a score that is
slightly, invisibly, optimistic.


```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")

print("average precision per fold:", scores.round(3))
print(f"mean {scores.mean():.3f} ± {scores.std():.3f}")
```

    average precision per fold: [0.383 0.353 0.405 0.312 0.365]
    mean 0.363 ± 0.031


The same fold spread as session 6, now computed honestly by construction rather
than by care.

### Preprocessing choices become tunable

Once preprocessing lives inside the pipeline, `GridSearchCV` can search it. The
`__` notation reaches into nested steps: `prepare__numeric__impute__strategy` is
"the `strategy` of the `impute` step of the `numeric` branch of `prepare`".


```python
from sklearn.model_selection import GridSearchCV

grid = {
    "prepare__numeric__impute__strategy": ["median", "mean"],
    "classify__C": [0.1, 1, 10],
}

search = GridSearchCV(model, grid, cv=cv, scoring="average_precision", n_jobs=-1)
search.fit(X_train, y_train)

print("best:", search.best_params_)
print(f"best score: {search.best_score_:.4f}   (baseline {scores.mean():.4f})")
```

    best: {'classify__C': 0.1, 'prepare__numeric__impute__strategy': 'median'}
    best score: 0.3641   (baseline 0.3634)


"Should I impute with the median or the mean?" stops being an opinion and becomes
a measurement — one that is made inside each fold, where it belongs.

---

## 3. Engineered features, inside the pipeline

Session 3's ratios were computed on the whole dataframe before splitting. That
was safe because a ratio of two columns in the same row uses no information from
other rows — but only just, and the habit is dangerous.

Anything computed from a column belongs inside the pipeline. `FunctionTransformer`
wraps a plain function into an estimator so it can sit in the chain.


```python
from sklearn.preprocessing import FunctionTransformer


def add_credit_ratios(frame):
    """Add the ratios a credit officer computes by hand. Row-wise only."""
    frame = frame.copy()
    income = frame["annual_income"].fillna(frame["annual_income"].median())
    frame["loan_to_income"] = frame["loan_amount"] / income
    frame["monthly_instalment"] = frame["loan_amount"] / frame["tenure_months"]
    return frame


NUMERIC_PLUS = NUMERIC + ["loan_to_income", "monthly_instalment"]

featured_model = Pipeline([
    ("engineer", FunctionTransformer(add_credit_ratios)),
    ("prepare", ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC_PLUS),
        ("categorical", categorical_steps, CATEGORICAL),
    ])),
    ("classify", LogisticRegression(max_iter=1000)),
])

featured_scores = cross_val_score(featured_model, X_train, y_train, cv=cv,
                                  scoring="average_precision")
print(f"without ratios {scores.mean():.4f} ± {scores.std():.4f}")
print(f"with ratios    {featured_scores.mean():.4f} ± {featured_scores.std():.4f}")
```

    without ratios 0.3634 ± 0.0313
    with ratios    0.3624 ± 0.0306


Still no improvement, exactly as session 3 measured — but now the feature step
travels with the model. When session 9 serves this object, the ratios are computed
inside it, from the raw fields the API receives. Nobody has to remember to
recreate them.

That is the real argument for putting feature engineering in the pipeline: **the
training code and the serving code become the same code.** Mismatch between those
two is one of the most common production failures in the industry, and it is
entirely preventable.

---

## 4. Reproducibility: what actually has to be recorded

"Reproducible" means someone else, on another machine, in six months, gets your
number. That needs five things, and a notebook records none of them by default.

| What | Why | How |
|------|-----|-----|
| **Code version** | The pipeline definition changes weekly | Git commit hash |
| **Data version** | "The extract" is not a fixed thing | File hash, row count, date range |
| **Parameters** | Which C? Which threshold? Which imputer? | Logged with the run |
| **Metrics** | The claim being made | Logged with the run |
| **Environment** | scikit-learn changes defaults between versions | `requirements.txt`, Python version |

Plus one thing that is not a record but a habit: **set every seed**. Our
`random_state=42` appears in the split and in every estimator that samples. Without
it, two runs of the same code produce two different numbers and you cannot tell
improvement from noise.


```python
import hashlib
import sklearn

data_file = ROOT / "data" / "loan_default.csv"
data_hash = hashlib.md5(data_file.read_bytes()).hexdigest()[:12]

run_context = {
    "data_file": data_file.name,
    "data_md5": data_hash,
    "rows": len(loans),
    "date_range": f"{loans['application_date'].min():%Y-%m-%d} to {loans['application_date'].max():%Y-%m-%d}",
    "sklearn_version": sklearn.__version__,
    "random_state": 42,
}
for key, value in run_context.items():
    print(f"  {key:<16} {value}")
```

      data_file        loan_default.csv
      data_md5         18b4b7c4918f
      rows             12000
      date_range       2023-01-01 to 2024-12-31
      sklearn_version  1.9.0
      random_state     42


That dictionary is the minimum. If the data hash changes, the number changes, and
knowing which happened first saves a week of confusion.

---

## 5. Experiment tracking with MLflow

By session 6 we had run dozens of configurations. Which one produced the 0.3665?
What threshold was it evaluated at? Where is that model now?

If the answer is "somewhere in the notebook", the work is not recoverable. An
experiment tracker is a database that answers those questions automatically.

**MLflow** runs entirely on your laptop. We point it at a local SQLite file — the
old `./mlruns` file store still works in some versions but is on its way out, and
SQLite is what the tool now expects.


```python
import logging

import mlflow
import mlflow.sklearn

logging.getLogger("mlflow").setLevel(logging.ERROR)   # keep the notebook output readable

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("loan-default")

print("tracking:", mlflow.get_tracking_uri())
print("\nTo browse these runs, from the repository root run:")
print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("  (then open http://127.0.0.1:5000)")
```

    tracking: sqlite:////Users/aayush/Documents/Kings/AI/AIModule/mlflow.db
    
    To browse these runs, from the repository root run:
      mlflow ui --backend-store-uri sqlite:///mlflow.db
      (then open http://127.0.0.1:5000)


A **run** is one training attempt. Inside a run you log three kinds of thing:

- **parameters** — the settings you chose (`log_param`)
- **metrics** — the numbers you measured (`log_metric`)
- **artifacts** — files, including the model itself (`log_model`)


```python
from sklearn.ensemble import RandomForestClassifier

candidates = {
    "logistic-C1": Pipeline([("prepare", preprocessing),
                             ("classify", LogisticRegression(max_iter=1000))]),
    "logistic-C0.1": Pipeline([("prepare", preprocessing),
                               ("classify", LogisticRegression(max_iter=1000, C=0.1))]),
    "random-forest": Pipeline([("prepare", preprocessing),
                               ("classify", RandomForestClassifier(n_estimators=200,
                                                                   random_state=42))]),
}

for run_name, candidate in candidates.items():
    with mlflow.start_run(run_name=run_name):
        fold_scores = cross_val_score(candidate, X_train, y_train, cv=cv,
                                      scoring="average_precision")
        candidate.fit(X_train, y_train)
        test_probabilities = candidate.predict_proba(X_test)[:, 1]

        mlflow.log_params(run_context)
        mlflow.log_param("estimator", candidate.named_steps["classify"].__class__.__name__)
        mlflow.log_metric("cv_average_precision", fold_scores.mean())
        mlflow.log_metric("cv_std", fold_scores.std())
        mlflow.log_metric("test_average_precision",
                          average_precision_score(y_test, test_probabilities))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, test_probabilities))

        mlflow.sklearn.log_model(candidate, name="model", serialization_format="pickle")

        print(f"logged {run_name}: cv {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
```

    logged logistic-C1: cv 0.3634 ± 0.0313


    logged logistic-C0.1: cv 0.3641 ± 0.0317


    logged random-forest: cv 0.3340 ± 0.0258


Three runs, three models saved, every parameter and metric recorded — in about
fifteen lines wrapped around code we had already written.

(MLflow warns that pickle can execute code when loaded. For a local teaching
setup that is fine; in a shared production system you would use the safer `skops`
format or keep the artifact store locked down.)


```python
runs = mlflow.search_runs(experiment_names=["loan-default"])

comparison = runs[["tags.mlflow.runName", "params.estimator",
                   "metrics.cv_average_precision", "metrics.test_average_precision",
                   "metrics.test_roc_auc"]]
comparison.columns = ["run", "estimator", "cv_AP", "test_AP", "test_AUC"]
print(comparison.round(4).to_string(index=False))
```

              run              estimator  cv_AP  test_AP  test_AUC
    random-forest RandomForestClassifier 0.3340   0.3672    0.7628
    logistic-C0.1     LogisticRegression 0.3641   0.3807    0.7819
      logistic-C1     LogisticRegression 0.3634   0.3817    0.7823


That table is the thing a notebook cannot give you: every experiment you have
ever run, side by side, with the settings that produced each one — queryable from
Python, and browsable in `mlflow ui`.

Note the disagreement in that table. On **cross-validated** average precision the
logistic runs win comfortably (0.364 against 0.334); on the **single test split**
the random forest looks competitive. Session 6 told us which to believe — the
cross-validated number, averaged over five folds, rather than one draw.

Without a tracker that comparison lives in somebody's memory of "I think the
forest was better". With one, it is a table anybody can re-read in six months.

---

## 6. Model versioning: getting a run's model back

Every logged model has a URI. Load it and it is the pipeline again — preprocessing
included, ready to score raw applications.


```python
best = runs.sort_values("metrics.cv_average_precision", ascending=False).iloc[0]
best_name = best["tags.mlflow.runName"]

loaded_model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")

one_application = X_test.head(1)
print(f"best run: {best_name}  (cv AP {best['metrics.cv_average_precision']:.4f})")
print(f"reloaded from MLflow: {loaded_model.predict_proba(one_application)[0, 1]:.4f}")
print(f"still in memory here: {candidates[best_name].predict_proba(one_application)[0, 1]:.4f}")
```

    best run: logistic-C0.1  (cv AP 0.3641)
    reloaded from MLflow: 0.2828
    still in memory here: 0.2828


Same number, from a model reconstructed out of the tracking store rather than from
memory. That round trip is the whole point: **the run is the deliverable, not the
notebook.**

In a larger setup the next step is the **model registry** — the same store, with
named models and stages (`staging`, `production`) so a service can ask for "the
current production loan-default model" rather than a run id. The mechanics are the
same; the discipline is the same; only the naming changes. Session 8 puts that in
the context of the wider MLOps lifecycle.

---

## 7. Rules that survive contact with a real project

1. **Everything that learns from data goes inside the pipeline.** Imputers,
   scalers, encoders, feature builders, resamplers. If it has a `fit`, it belongs
   in the chain.
2. **Nothing is fitted outside cross-validation.** If you cannot express a step as
   a pipeline stage, that is a warning sign, not an excuse.
3. **One run, one record.** Parameters, metrics, data hash, code version, model
   artifact — logged at the moment they are true, not reconstructed later.
4. **Name runs like a human.** `logistic-C0.1-with-ratios` beats
   `run_2026_09_16_17_42`.
5. **Log the failures too.** The tracker's value is the record of what did *not*
   work, which is how you stop a colleague repeating it next month.
6. **`imbalanced-learn` has its own `Pipeline`.** `imblearn.pipeline.Pipeline`
   resamples the training fold only, and leaves validation folds alone.
   scikit-learn's plain `Pipeline` cannot do that correctly — if you use SMOTE,
   use theirs.

---

## Your turn

**1. Add the threshold to the pipeline story.** Our model outputs probabilities;
the decision needs a threshold. Log the chosen threshold as a parameter and the
resulting recall, flag rate and expected cost as metrics, for thresholds 0.15,
0.20 and 0.25. Which run would you promote?

**2. Search preprocessing properly.** Extend the grid to include
`prepare__categorical__encode__drop` (`"first"` versus `None`) and
`prepare__numeric__scale` (`StandardScaler()` versus `"passthrough"`). Does
anything beat the defaults?

**3. Use the imblearn pipeline.** Rebuild the model with
`imblearn.pipeline.Pipeline` and a `SMOTE` step, cross-validate it, and log the
run. Compare its cross-validated average precision to the plain pipeline's. Does
resampling help once it is done correctly inside each fold?

**4. Break reproducibility on purpose.** Remove `random_state` from the split, run
the same training three times, and log all three. How far apart are the metrics?
Is the difference bigger than the gap between your best two models?

**5. Size-aware thresholds, tracked.** Session 6's size-aware thresholds saved
NPR 8.6m. Log that configuration as a run with its cost metric, so the saving is
recorded rather than remembered.

**6. Open the UI.** Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` from
the repository root, sort your runs by `cv_average_precision`, and find the exact
parameters of the best one. This is the workflow you will use for the final project.

---

## If you remember nothing else

**A pipeline is one estimator from raw columns to prediction.** Anything that
learns from data lives inside it — then leakage is structurally impossible rather
than carefully avoided.

**`ColumnTransformer` handles the two-branch reality of tabular data**: impute and
scale the numbers, impute and encode the text, in one object.

**Cross-validating a pipeline refits preprocessing per fold.** That is the honest
number; doing it by hand is quietly optimistic.

**Training code and serving code should be the same code.** Feature engineering
inside the pipeline is how that happens.

**A run must record five things**: code version, data version, parameters,
metrics, environment. Set your seeds so the sixth — luck — is not a factor.

**The run is the deliverable, not the notebook.** If you cannot reload a model and
reproduce its number, you have a demo, not a system.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Pipeline** | Transformers plus a final estimator composed into one estimator |
| **`ColumnTransformer`** | Applies different preprocessing to different column groups |
| **`FunctionTransformer`** | Wraps a plain function as a pipeline step |
| **`handle_unknown="ignore"`** | Encoder behaviour for categories unseen in training |
| **`__` notation** | How grid search addresses nested pipeline parameters |
| **Experiment** | A named collection of runs in a tracker |
| **Run** | One training attempt, with its parameters, metrics and artifacts |
| **Artifact** | A file produced by a run — the model, a plot, a data sample |
| **Model URI** | The address of a logged model, e.g. `runs:/<run_id>/model` |
| **Model registry** | Named, versioned models with lifecycle stages |
| **Reproducibility** | Same code, data, parameters and environment giving the same number |

## Further reading

- scikit-learn User Guide, *Pipelines and composite estimators* — and the
  *Common pitfalls* page on why pipelines prevent leakage.
- MLflow documentation, *MLflow Tracking* quickstart — ten minutes end to end.
- `imbalanced-learn` documentation, *Pipeline* — the resampling-aware version.
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* — glue code
  and pipeline jungles, which is precisely what this session is defending against.

---

**Next session:** *Introduction to MLOps* — the lifecycle around the model:
deployment workflow, monitoring, CI/CD concepts, and the drift that has been
sitting in this dataset since session 2.
