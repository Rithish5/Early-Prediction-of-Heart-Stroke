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


# --- Custom Personalized Advice Function (Updated) ---
def display_personalized_advice(chol, trestbps, fbs):
    """Generates advice based on specific user input values for high-risk cases."""
    
    st.markdown("---")
    st.markdown("<h3 style='color: #B22222; text-align: center;'>🚨 Personalized Cardiac Health Recommendations</h3>", unsafe_allow_html=True)
    st.markdown("**Consultation is Mandatory.** Given the high-risk prediction, seek immediate consultation with a cardiologist.", unsafe_allow_html=True)

    st.markdown("#### Focus Areas:")
    
    # 1. Cholesterol Advice
    if chol > 200:
        st.error(f"**High Cholesterol ({chol} mg/dL):** Your level exceeds the recommended limit. Immediately focus on a low-fat, low-cholesterol diet rich in fiber. Limit red meats and processed foods.")
    elif chol > 160:
        st.warning(f"**Elevated Cholesterol ({chol} mg/dL):** While not critically high, this level contributes to risk. Maintain a healthy, heart-friendly diet and increase omega-3 fatty acids.")
    
    # 2. Blood Pressure Advice
    if trestbps >= 140:
        st.error(f"**High Blood Pressure ({trestbps} mmHg):** This level is categorized as Hypertension. Consult your doctor about medication and drastically reduce sodium (salt) intake to under 1,500 mg/day.")
    elif trestbps >= 120 and trestbps < 140:
        st.warning(f"**Elevated Blood Pressure ({trestbps} mmHg):** This is considered elevated. Implement stress management techniques and consistent aerobic exercise.")

    # 3. Fasting Blood Sugar (Diabetes) Advice
    if fbs == 1:
        st.error(f"**High Fasting Blood Sugar (> 120 mg/dL):** This indicates a potential pre-diabetic or diabetic state. Strictly control carbohydrate intake and monitor blood sugar levels daily.")
    else:
        st.info("**Fasting Blood Sugar:** Currently within a lower-risk range.")

    st.markdown("---")
    st.markdown("**General Action Plan:** Aim for 30 minutes of moderate exercise daily (after medical clearance) and stop smoking immediately if applicable.")

# --- Streamlit UI Setup with ADVANCED STYLING ---

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
    }

    /* Custom Background (Subtle Gradient) */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #c9e8fb 100%); /* Light Blue Gradient */
    }

    /* Custom Header */
    .big-font {
        font-family: 'Roboto', sans-serif;
        font-size:36px !important;
        font-weight: 700;
        color: #004d99; /* Darker blue for contrast */
        text-align: center;
        padding-top: 15px;
        padding-bottom: 20px;
        border-bottom: 3px solid #0077b6;
        margin-bottom: 20px;
    }
    
    /* Input Form Container Card */
    .st-emotion-cache-12quk7f { /* This targets the main content wrapper */
        background-color: #ffffff; 
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }

    /* Result Box Styling (Kept strong) */
    .result-box {
        padding: 30px;
        border-radius: 15px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.3);
        background-color: #ffffff; /* Ensure white background for high contrast */
        transition: transform 0.3s ease;
    }
    .result-box:hover {
        transform: translateY(-5px);
    }
    .risk-score {
        font-family: 'Roboto', sans-serif;
        font-size: 56px;
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
    
    # --- DISPLAY PERSONALIZED ADVICE IF HIGH RISK ---
    if prediction == 1:
        # Pass the key input values to the personalized function
        display_personalized_advice(chol, trestbps, fbs)


st.markdown("---")
st.caption("Disclaimer: This ML prediction tool is for informational and educational purposes only. Always consult a qualified healthcare professional for medical advice.")
