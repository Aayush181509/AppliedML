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
