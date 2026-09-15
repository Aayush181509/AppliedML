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
