import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("mlr_model.pkl", "rb"))

st.title("💰 Investment Profit Predictor")

st.write("Enter the investment details to predict profit")

# Numeric inputs
digital = st.number_input("Digital Marketing Spend", min_value=0.0)
promotion = st.number_input("Promotion Spend", min_value=0.0)
research = st.number_input("Research Spend", min_value=0.0)

# Categorical input
state = st.selectbox(
    "State",
    ["Bangalore", "Chennai", "Hyderabad"]
)

# One-hot encoding for State (MUST match backend)
if state == "Bangalore":
    state_bangalore, state_chennai, state_hyderabad = 1, 0, 0
elif state == "Chennai":
    state_bangalore, state_chennai, state_hyderabad = 0, 1, 0
else:
    state_bangalore, state_chennai, state_hyderabad = 0, 0, 1

# Final input array (6 features)
input_data = np.array([[
    digital,
    promotion,
    research,
    state_bangalore,
    state_chennai,
    state_hyderabad
]])

# Predict button
if st.button("Predict Profit"):
    prediction = model.predict(input_data)
    st.success(f"📈 Predicted Profit: ₹ {prediction[0]:,.2f}")