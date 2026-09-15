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
