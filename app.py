import streamlit as st
import pandas as pd
import pickle
import os

# --- CONFIGURATION ---
MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'

# Set the page title/icon
st.set_page_config(page_title="Early Heart Stroke Prediction App", layout="centered")

# --- SEO METADATA INJECTION ---
def inject_seo_tags():
    """Injects meta tags for search engine optimization (SEO)."""
    st.markdown(
        f"""
        <head>
            <meta name="description" content="An advanced machine learning tool for early prediction and risk analysis of heart stroke using clinical data. Developed by Rithish5.">
            <meta name="keywords" content="heart stroke prediction, cardiovascular risk, machine learning, medical app, Streamlit, data science, stroke risk analyzer">
            <meta property="og:title" content="Early Heart Stroke Prediction App">
            <meta property="og:description" content="Predict heart stroke risk instantly using an ML model based on 13 clinical features.">
        </head>
        """,
        unsafe_allow_html=True,
    )

# Inject the SEO tags at the very start
inject_seo_tags()
# --- END SEO INJECTION ---


# --- Load Model and Scaler ---
model = None
scaler = None

try:
    # Check if files exist before trying to open
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        st.error("Error: Model files ('model.pkl' and 'scaler.pkl') not found.")
        st.warning("Please run trained_model.py to generate these files.")
        st.stop()
        
    with open(MODEL_FILE, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_FILE, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()


# --- Custom Advice Function (Basic, non-personalized version) ---
def display_basic_advice():
    """Displays simple advice for high-risk patients."""
    st.markdown("---")
    st.markdown("<h3 style='color: #B22222; text-align: center;'>⚠️ Crucial Lifestyle Advice</h3>", unsafe_allow_html=True)
    st.markdown("""
    **Consultation is Mandatory.** Given the high-risk prediction, seek immediate consultation with a cardiologist.
    
    * **Diet:** Focus on a Mediterranean-style diet. Drastically reduce sodium (salt) intake to under 1,500 mg/day. Limit saturated fats and added sugars.
    * **Activity:** Aim for at least 30 minutes of moderate-intensity exercise (like brisk walking) most days of the week, after medical clearance.
    * **Monitoring:** Regularly monitor blood pressure and blood sugar levels.
    """)


# --- Streamlit UI Setup ---

st.markdown("""
    <style>
    .big-font {
        font-size:32px !important;
        font-weight: bold;
        color: #0077b6; /* Changed color for distinction */
        text-align: center;
        padding-bottom: 10px;
    }
    .result-box {
        padding: 25px;
        border-radius: 12px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .risk-score {
        font-size: 48px;
        font-weight: 900;
        margin: 10px 0;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Early Heart Stroke Prediction App</p>', unsafe_allow_html=True)
st.write("🏥 Enter the patient's clinical parameters below.")


# --- Input Form ---

# Feature column order: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider('Age', 20, 100, 50, help="Age in years.")
    sex = st.selectbox('Sex', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1])[0]
    cp = st.selectbox('Chest Pain Type (cp)', options=[
        (0, 'Asymptomatic (0)'), (1, 'Typical Angina (1)'), 
        (2, 'Atypical Angina (2)'), (3, 'Non-Anginal Pain (3)')
    ], format_func=lambda x: x[1])[0]

with col2:
    trestbps = st.number_input('Resting BP (trestbps)', 80, 200, 120)
    chol = st.number_input('Cholesterol (chol)', 100, 600, 240)
    fbs = st.selectbox('Fasting Sugar > 120 (fbs)', options=[(1, 'True (1)'), (0, 'False (0)')], format_func=lambda x: x[1])[0]
    
with col3:
    restecg = st.selectbox('Resting ECG (restecg)', options=[
        (0, 'Normal (0)'), (1, 'ST-T wave abnormality (1)'), 
        (2, 'LV Hypertrophy (2)')
    ], format_func=lambda x: x[1])[0]
    thalachh = st.number_input('Max Heart Rate (thalachh)', 70, 220, 150)
    exang = st.selectbox('Exercise Angina (exang)', options=[(1, 'Yes (1)'), (0, 'No (0)')], format_func=lambda x: x[1])[0]

# New Row for remaining features
col4, col5, col6 = st.columns(3)

with col4:
    oldpeak = st.number_input('ST Depression (oldpeak)', 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox('Slope of ST Segment (slope)', options=[
        (0, 'Upsloping (0)'), (1, 'Flat (1)'), (2, 'Downsloping (2)')
    ], format_func=lambda x: x[1])[0]

with col5:
    ca = st.selectbox('Major Vessels (ca)', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia (thal)', options=[
        (1, 'Normal (1)'), (2, 'Fixed Defect (2)'), (3, 'Reversible Defect (3)')
    ], format_func=lambda x: x[1])[0]

# --- Prediction Logic ---
st.markdown("---")

if st.button('Analyze Patient Risk', use_container_width=True, type="primary"):
    
    # 1. Create a DataFrame from user inputs
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalachh, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # 2. Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # 3. Make prediction
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    # Calculate Risk Score (Probability of having heart disease)
    risk_score = prediction_proba[1] * 100
    
    # 4. Display Result
    if prediction == 1:
        risk_level = "High Risk"
        color = "#B22222" # Red
        score_text = f"{risk_score:.0f}%"
    else:
        risk_level = "Low Risk"
        color = "#4CAF50" # Green
        # Display probability of NO disease for low risk
        score_text = f"{(100 - risk_score):.0f}%"

    st.markdown(f"""
        <div class="result-box" style="background-color: #ffffff; border: 3px solid {color};">
            <h3 style="color: {color}; margin-bottom: 5px;">{risk_level} Prediction</h3>
            <span class="risk-score" style="color: {color};">{score_text}</span>
            <p style="font-size: 16px; margin-top: 5px;">
                { 'Based on the features, the model predicts a high probability of heart disease.' if prediction == 1 else 'Based on the features, the model predicts a low probability of heart disease.' }
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- DISPLAY BASIC ADVICE IF HIGH RISK ---
    if prediction == 1:
        display_basic_advice()


st.markdown("---")
st.caption("Disclaimer: This ML prediction tool is for informational and educational purposes only. Always consult a qualified healthcare professional for medical advice.")
