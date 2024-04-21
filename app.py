import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Function to load pre-saved objects (model and scaler)
def load_model_scaler():
    model = {
        'Linear Regression': joblib.load('models/linear_regression_model.pkl'),
        'Random Forest': joblib.load('models/random_forest_model.pkl'),
        'Gradient Boosting': joblib.load('models/gradient_boosting_model.pkl')
    }
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

models, scaler = load_model_scaler()

# Function to make prediction
def make_prediction(input_data, model_name):
    # Transform inputs using the loaded scaler
    input_scaled = scaler.transform(input_data)
    # Predict
    model = models[model_name]
    prediction = model.predict(input_scaled)
    return prediction

def user_input_features():
    st.sidebar.header('User Input Features for Air Quality Prediction')
    # Define your input sliders
    co = st.sidebar.slider('CO(GT) level (mg/m³)', 0.1, 11.9, 2.0)
    benzene = st.sidebar.slider('C6H6(GT) level (µg/m³)', 0.1, 63.7, 5.0)
    nmhc = st.sidebar.slider('NMHC(GT) level (ppb)', 7.0, 1189.0, 100.0)
    nox = st.sidebar.slider('NOx(GT) level (ppb)', 2.0, 1479.0, 150.0)
    no2 = st.sidebar.slider('NO2(GT) level (ppb)', 2.0, 340.0, 50.0)
    pt08_s1_co = st.sidebar.slider('PT08.S1(CO) Tin oxide level', 647.0, 2040.0, 1000.0)
    pt08_s3_nox = st.sidebar.slider('PT08.S3(NOx) Tungsten oxide level', 322.0, 2683.0, 900.0)
    pt08_s4_no2 = st.sidebar.slider('PT08.S4(NO2) Tungsten oxide level', 551.0, 2775.0, 1500.0)
    pt08_s5_o3 = st.sidebar.slider('PT08.S5(O3) Indium oxide level', 221.0, 2523.0, 900.0)
    temp = st.sidebar.slider('Temperature (°C)', -1.9, 44.6, 20.0)
    rh = st.sidebar.slider('Relative Humidity (%)', 9.2, 88.7, 50.0)
    ah = st.sidebar.slider('Absolute Humidity', 0.1847, 2.231, 1.0)
    
    # Collect all inputs into a DataFrame
    input_data = pd.DataFrame([[co, benzene, nmhc, nox, no2, pt08_s1_co, pt08_s3_nox, pt08_s4_no2, pt08_s5_o3, temp, rh, ah]],
                              columns=['CO(GT)', 'C6H6(GT)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S1(CO)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'])
    return input_data

def main():
    st.title('Air Quality Prediction Interface')
    st.write("""
    This interface predicts the level of Non-Methane Hydrocarbons (NMHC) based on various air quality indicators.
    Adjust the sliders to change input values and see how they affect the predicted NMHC levels.
    """)

    input_df = user_input_features()

    model_choice = st.sidebar.selectbox('Choose Model', list(models.keys()))
    prediction_output = st.empty()  # Placeholder for dynamic output
    prediction = make_prediction(input_df, model_choice)

    prediction_output.write(f'You selected model: **{model_choice}**')
    prediction_output.header('Predicted NMHC Level:')
    prediction_output.write(f'**{prediction[0]:.4f}** micrograms/m³')

    st.write('Input Features:', input_df)
    
   

if __name__ == '__main__':
    main()
