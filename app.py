
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from feature_engineering import create_features


# 1. Load Model

st.set_page_config(
    page_title="Mobile Wallet Fraud Detection",
    layout="wide"
)

@st.cache_resource
def load_model():
    bundle = joblib.load("fraud_detection_model.joblib")
    return bundle["model"], bundle["threshold"]


model, saved_threshold = load_model()


# 2. Application Header

st.title(" Mobile Wallet Fraud Detection")

st.write(
    "An ML-powered application for identifying "
    "potentially fraudulent mobile wallet transactions."
)

st.info(
    "The model provides risk estimates to support "
    "fraud investigation. It does not guarantee fraud detection."
)


# 3. Threshold and Cost Settings

st.sidebar.header("Detection Settings")

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(saved_threshold),
    step=0.01
)

st.sidebar.caption(
    "Transactions with a fraud probability at or above "
    "this threshold are flagged."
)

st.sidebar.subheader("Estimated Fraud Costs")

FN_COST = st.sidebar.number_input(
    "Cost of Missing One Fraud Transaction",
    min_value=0,
    value=1000,
    step=100
)

FP_COST = st.sidebar.number_input(
    "Cost of One False Alarm",
    min_value=0,
    value=10,
    step=1
)


# 4. Input Data

st.header("Transaction Analysis")

input_method = st.radio(
    "Choose Input Method",
    ["Single Transaction", "CSV Upload"],
    horizontal=True
)


required_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]


# 5. Prediction Function

def predict_transactions(data):

    probabilities = model.predict_proba(data)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = data.copy()

    results["Fraud Probability"] = probabilities
    results["Prediction"] = predictions

    results["Prediction Label"] = results["Prediction"].map({
        0: "Likely Legitimate",
        1: "Flag for Review"
    })

    return results


# 6. Single Transaction Form

if input_method == "Single Transaction":

    with st.form("transaction_form"):

        st.subheader("Transaction Information")

        step = st.number_input(
            "Transaction Step",
            min_value=1,
            max_value=744,
            value=1
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        st.subheader("Origin Account")

        oldbalanceOrg = st.number_input(
            "Origin Balance Before Transaction",
            min_value=0.0,
            value=5000.0
        )

        st.subheader("Destination Account")

        oldbalanceDest = st.number_input(
            "Destination Balance Before Transaction",
            min_value=0.0,
            value=2000.0
        )

        submitted = st.form_submit_button(
            "Analyze Transaction"
        )

    if submitted:

        transaction = pd.DataFrame([{
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "oldbalanceDest": oldbalanceDest
        }])

        results = predict_transactions(transaction)

        probability = results["Fraud Probability"].iloc[0]
        prediction = results["Prediction"].iloc[0]

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        if prediction == 1:
            st.error("Suspicious Transaction — Flag for Review")
        else:
            st.success("Likely Legitimate Transaction")

        st.write(
            f"Current threshold: {threshold:.2f}"
        )


# 7. CSV Upload

else:

    st.subheader("Upload Transaction Dataset")

    st.write(
        "Upload a CSV file containing the five required "
        "transaction input features. An optional isFraud "
        "column can be included for model evaluation."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_data = pd.read_csv(uploaded_file)

        missing_columns = [
            col for col in required_columns
            if col not in uploaded_data.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

        elif uploaded_data.empty:

            st.error("The uploaded CSV file is empty.")

        elif uploaded_data[required_columns].isnull().any().any():

            st.error(
                "Please handle missing values in the required "
                "input columns before uploading."
            )

        else:

            st.write("Uploaded Dataset Preview")

            st.dataframe(uploaded_data.head())

            if st.button("Analyze All Transactions"):

                results = predict_transactions(
                    uploaded_data[required_columns].copy()
                )

                st.session_state["results"] = results
                st.session_state["uploaded_data"] = uploaded_data


# 8. Results and Cost Analysis

if "results" in st.session_state:

    results = st.session_state["results"].copy()

    results["Prediction"] = (
        results["Fraud Probability"] >= threshold
    ).astype(int)

    results["Prediction Label"] = results["Prediction"].map({
        0: "Likely Legitimate",
        1: "Flag for Review"
    })

    st.divider()
    st.header("Transaction Results")

    st.dataframe(
        results,
        use_container_width=True
    )

    st.download_button(
        "Download Prediction Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )

    # Ground-truth evaluation is only possible
    # if the uploaded CSV contains isFraud.
    if "isFraud" in st.session_state.get(
        "uploaded_data", pd.DataFrame()
    ).columns:

        original_data = st.session_state["uploaded_data"]

        y_true = original_data["isFraud"].reset_index(drop=True)

        if y_true.isin([0, 1]).all():

            y_pred = results["Prediction"].reset_index(drop=True)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            ).ravel()

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            pr_auc = average_precision_score(
                y_true,
                results["Fraud Probability"]
            )

            missed_cost = fn * FN_COST
            false_alarm_cost = fp * FP_COST

            total_cost = (
                fn * FN_COST +
                fp * FP_COST
            )

            st.header("Fraud Detection Performance")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Detected Fraud (TP)", tp)
            col2.metric("Missed Fraud (FN)", fn)
            col3.metric("False Alarms (FP)", fp)
            col4.metric("Correct Legitimate (TN)", tn)

            col1, col2, col3 = st.columns(3)

            col1.metric("Precision", f"{precision:.4f}")
            col2.metric("Recall", f"{recall:.4f}")
            col3.metric("F1-Score", f"{f1:.4f}")

            st.metric("PR-AUC", f"{pr_auc:.4f}")

            st.subheader("Estimated Cost Analysis")

            col1, col2, col3 = st.columns(3)

            col1.metric("Missed Fraud Cost", f"{missed_cost:,}")
            col2.metric("False Alarm Cost", f"{false_alarm_cost:,}")
            col3.metric("Total Estimated Cost", f"{total_cost:,}")

            st.caption(
                "Costs are hypothetical estimates based on the values "
                "configured in the sidebar, not actual financial losses."
            )

        else:

            st.warning(
                "The isFraud column must contain only 0 and 1 "
                "to calculate evaluation metrics."
            )

    else:

        st.warning(
            "To calculate missed fraud, false alarms, and their "
            "costs, upload a labeled CSV containing an isFraud "
            "column. Without actual labels, these metrics cannot "
            "be determined."
        )


# 9. Footer

st.divider()

st.caption(
    "Mobile Wallet Fraud Detection | "
    "Model trained using the synthetic PaySim dataset."
)
