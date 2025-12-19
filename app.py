import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction (ANN)")
st.write("Predict whether a customer is likely to churn.")

# =============================
# Load model & artifacts
# =============================
@st.cache_resource
def load_artifacts():
    # Load trained ANN model
    model = load_model("model.h5")

    # Load encoders & scaler
    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder = pickle.load(f)

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Get feature order directly from scaler
    feature_order = scaler.feature_names_in_.tolist()

    return model, onehot_encoder, label_encoder_gender, scaler, feature_order


model, onehot_encoder, label_encoder_gender, scaler, feature_order = load_artifacts()

# =============================
# User Input Section
# =============================
st.subheader("Enter Customer Details")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100, value=40)
Tenure = st.slider("Tenure (years)", 0, 10, 3)
Balance = st.number_input("Account Balance", value=5000.0)
NumOfProducts = st.slider("Number of Products", 1, 4, 2)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", value=5000.0)

# =============================
# Prediction Function
# =============================
def predict_churn(input_data):
    df = pd.DataFrame([input_data])

    # Encode Gender
    df["Gender"] = label_encoder_gender.transform(df["Gender"])

    # Encode Geography
    geo_encoded = onehot_encoder.transform(df[["Geography"]]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder.get_feature_names_out(["Geography"])
    )

    # Merge encoded features
    df = df.drop("Geography", axis=1)
    df = pd.concat([df.reset_index(drop=True), geo_df], axis=1)

    # Ensure correct feature order
    df = df[feature_order]

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prob = model.predict(df_scaled, verbose=0)[0][0]
    return prob


# =============================
# Predict Button
# =============================
if st.button("🔍 Predict Churn"):
    input_data = {
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }

    churn_prob = predict_churn(input_data)

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{churn_prob:.2%}")

    if churn_prob >= 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is NOT likely to churn")
