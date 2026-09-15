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
