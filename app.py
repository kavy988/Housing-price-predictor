import streamlit as st
import pandas as pd
import joblib

# Import must happen BEFORE loading the model, so joblib can find the classes
from custom_transformers import ClusterSimilarity, column_ratio, ratio_name, ratio_pipeline

st.title("California Housing Price Predictor")
st.write("Enter district details to predict the median house value.")

# Load the trained pipeline (preprocessing + model, all in one)
model = joblib.load("final_model.pkl")

# --- User inputs ---
longitude = st.number_input("Longitude", value=-122.0, format="%.2f")
latitude = st.number_input("Latitude", value=37.5, format="%.2f")
housing_median_age = st.slider("Housing Median Age", 1, 52, 30)
total_rooms = st.number_input("Total Rooms", value=2000)
total_bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=400)
median_income = st.number_input("Median Income (in $10,000s)", value=3.5, format="%.4f")
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Median House Value: ${prediction:,.2f}")
