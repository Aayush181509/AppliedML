"""Streamlit interface for the loan default service.

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
