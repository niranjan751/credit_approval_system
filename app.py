import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("credit_approval_model.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Card Approval Predictor")

st.write(
    "Enter the applicant information below to estimate "
    "whether the credit-card application is likely to be approved."
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.subheader("Applicant Information")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT COLUMN
# ------------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["a", "b"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        value=30.0
    )

    debt = st.number_input(
        "Debt",
        min_value=0.0,
        value=5.0
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["u", "y", "l", "t"]
    )

    bank_customer = st.selectbox(
        "Bank Customer",
        ["g", "p", "gg"]
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "c", "d", "cc", "i", "j", "k", "m",
            "r", "q", "w", "x", "e", "aa", "ff"
        ]
    )

    ethnicity = st.selectbox(
        "Ethnicity",
        ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"]
    )


# ------------------------------------------------------------
# RIGHT COLUMN
# ------------------------------------------------------------

with col2:

    years_employed = st.number_input(
        "Years Employed",
        min_value=0.0,
        value=1.25
    )

    prior_default = st.selectbox(
        "Previous Credit Default",
        ["t", "f"]
    )

    employed = st.selectbox(
        "Currently Employed",
        ["t", "f"]
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        value=2
    )

    drivers_license = st.selectbox(
        "Driver's License",
        ["f", "t"]
    )

    citizen = st.selectbox(
        "Citizen",
        ["g", "p", "s"]
    )

    zip_code = st.number_input(
        "ZIP Code",
        min_value=0.0,
        value=200.0
    )

    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=100000.0
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Credit Approval",
    use_container_width=True
):

    # IMPORTANT:
    # These remain A1-A15 internally because that is
    # what the trained ML model expects.

    new_application = pd.DataFrame([{

        "A1": gender,
        "A2": age,
        "A3": debt,
        "A4": marital_status,
        "A5": bank_customer,
        "A6": education_level,
        "A7": ethnicity,
        "A8": years_employed,
        "A9": prior_default,
        "A10": employed,
        "A11": credit_score,
        "A12": drivers_license,
        "A13": citizen,
        "A14": zip_code,
        "A15": income

    }])


    # Make prediction
    prediction = model.predict(new_application)[0]

    probability = model.predict_proba(
        new_application
    )[0][1]

    probability_percent = probability * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "✅ APPLICATION LIKELY APPROVED"
        )

    else:

        st.error(
            "❌ APPLICATION LIKELY REJECTED"
        )


    st.metric(
        label="Approval Probability",
        value=f"{probability_percent:.2f}%"
    )


    # ========================================================
    # BUSINESS INTERPRETATION
    # ========================================================

    if probability >= 0.80:

        st.info(
            "High-confidence approval prediction. "
            "The application may be suitable for fast-track review."
        )

    elif probability >= 0.50:

        st.warning(
            "Moderate approval probability. "
            "The application may require additional manual review."
        )

    else:

        st.warning(
            "Low approval probability. "
            "The application may require further investigation."
        )


    # ========================================================
    # SHOW APPLICATION DETAILS
    # ========================================================

    with st.expander("View Application Details"):

        display_data = pd.DataFrame({
            "Field": [
                "Gender",
                "Age",
                "Debt",
                "Marital Status",
                "Bank Customer",
                "Education Level",
                "Ethnicity",
                "Years Employed",
                "Previous Credit Default",
                "Currently Employed",
                "Credit Score",
                "Driver's License",
                "Citizen",
                "ZIP Code",
                "Annual Income"
            ],

            "Value": [
                gender,
                age,
                debt,
                marital_status,
                bank_customer,
                education_level,
                ethnicity,
                years_employed,
                prior_default,
                employed,
                credit_score,
                drivers_license,
                citizen,
                zip_code,
                income
            ]
        })

        st.table(display_data)


    st.caption(
        "This is a machine-learning prediction and should not be "
        "treated as an automatic final credit decision."
    )