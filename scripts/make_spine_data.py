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
