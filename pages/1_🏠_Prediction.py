import streamlit as st
import pandas as pd
import pickle
import os

# --- CONFIGURATION ---
MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'

# --- Load Model and Scaler ---
model = None
scaler = None

# Streamlit App Title and Configuration 
st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="centered")

# Custom Advice Function
def display_advice(input_data):
    """
    Displays structured, doctor-style advice for high-risk patients, 
    personalized based on their specific input data.
    """
    
    # Extract key values for personalization
    chol = input_data['chol'].iloc[0]
    trestbps = input_data['trestbps'].iloc[0]
    fbs = input_data['fbs'].iloc[0]
    
    st.markdown("<h2 style='color: #B22222; text-align: center;'>🚨 Personalized Cardiac Health Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("Based on the data provided, immediate lifestyle changes and medical consultation are strongly advised to mitigate your elevated risk.")
    
    st.markdown("---")

    col_findings, col_diet = st.columns(2)

    with col_findings:
        st.subheader("🔍 Key Risk Factors Identified")
        st.markdown(f"**Cholesterol Management:** Your current cholesterol is **{chol:.0f} mg/dl**. This is a significant factor. Dietary changes are essential to reduce this level below 200 mg/dl.")
        
        if trestbps >= 140:
            st.markdown(f"**Blood Pressure:** Your resting blood pressure is **{trestbps:.0f} mm Hg**. This elevated reading requires daily monitoring and management through diet, exercise, and possibly medication.")
        else:
            st.markdown(f"**Blood Pressure:** Your BP is **{trestbps:.0f} mm Hg**. While this reading is acceptable, consistent monitoring remains critical.")
            
        if fbs == 1:
            st.markdown("⚠️ **Blood Sugar:** Fasting blood sugar is high (**>120 mg/dl**). This indicates pre-diabetic or diabetic conditions, dramatically increasing cardiac risk. Diet and medical attention are urgent.")
        
    with col_diet:
        st.subheader("🥗 Diet & Lifestyle Action Plan")
        
        st.markdown("""
        **1. Sodium (Salt) Reduction:**
        * Limit intake to **under 1,500 mg/day** (about 2/3 teaspoon total).
        * Avoid all high-sodium processed foods, frozen dinners, and restaurant meals.
        
        **2. Sugar and Fats:**
        * **Sugar:** Eliminate sugary drinks and focus on fiber-rich whole grains over refined carbohydrates.
        * **Fats:** Switch to heart-healthy fats (olive oil, avocado, salmon) and cut out trans fats and deep-fried foods.

        **3. Exercise:**
        * Start with light activity, aiming for **30 minutes of brisk walking** daily. Consistency is more important than intensity initially.
        """)
        
    st.markdown("---")
    st.error("🩺 **Next Step:** Do not delay consulting a specialist (Cardiologist) immediately. Bring these specific data points for a detailed health plan.")


try:
    # Check if user is logged in (from streamlit_app.py)
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.error("Access Denied. Please navigate to the Welcome page to log in.")
        st.stop()
        
    # Check if files exist before trying to open
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        st.error("Error: Model files ('model.pkl' and 'scaler.pkl') not found in the root directory.")
        st.warning("Please ensure you have run **`python trained_model.py`** to generate these files.")
        st.stop()
        
    with open(MODEL_FILE, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_FILE, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()


# --- Streamlit UI Setup ---

# Custom styling (unchanged)
st.markdown("""
    <style>
    .big-font {
        font-size:32px !important;
        font-weight: bold;
        color: #B22222; 
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
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Cardiovascular Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(f"**Current User:** {st.session_state.user_email}")
st.markdown("---")
st.write("🏥 Enter the patient's clinical parameters to predict the presence of heart disease (Target=1).")


# --- Input Form ---

# Feature column order for the model: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
col1, col2, col3 = st.columns(3)

# Initialize input_data placeholder outside the button block
input_data = None

with col1:
    age = st.slider('Age', 20, 100, 50, help="Age in years.")
    sex = st.selectbox('Sex', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1], help="1=Male, 0=Female")[0]
    cp = st.selectbox('Chest Pain Type (cp)', options=[
        (0, 'Asymptomatic (0)'), (1, 'Typical Angina (1)'), 
        (2, 'Atypical Angina (2)'), (3, 'Non-Anginal Pain (3)')
    ], format_func=lambda x: x[1], help="Chest pain type")[0]

with col2:
    trestbps = st.number_input('Resting BP (trestbps)', 80, 200, 120, help="Resting blood pressure (mm Hg)")
    chol = st.number_input('Cholesterol (chol)', 100, 600, 240, help="Serum cholestoral (mg/dl)")
    fbs = st.selectbox('Fasting Sugar > 120 (fbs)', options=[(1, 'True (1)'), (0, 'False (0)')], format_func=lambda x: x[1], help="Fasting blood sugar > 120 mg/dl")[0]
    
with col3:
    restecg = st.selectbox('Resting ECG (restecg)', options=[
        (0, 'Normal (0)'), (1, 'ST-T wave abnormality (1)'), 
        (2, 'LV Hypertrophy (2)')
    ], format_func=lambda x: x[1], help="Resting electrocardiographic results")[0]
    thalachh = st.number_input('Max Heart Rate (thalachh)', 70, 220, 150, help="Maximum heart rate achieved")
    exang = st.selectbox('Exercise Angina (exang)', options=[(1, 'Yes (1)'), (0, 'No (0)')], format_func=lambda x: x[1], help="Exercise induced angina")[0]

# New Row for remaining features
col4, col5, col6 = st.columns(3)

with col4:
    oldpeak = st.number_input('ST Depression (oldpeak)', 0.0, 6.2, 1.0, step=0.1, help="ST depression induced by exercise relative to rest")
    slope = st.selectbox('Slope of ST Segment (slope)', options=[
        (0, 'Upsloping (0)'), (1, 'Flat (1)'), (2, 'Downsloping (2)')
    ], format_func=lambda x: x[1], help="Slope of the peak exercise ST segment")[0]

with col5:
    ca = st.selectbox('Major Vessels (ca)', options=[0, 1, 2, 3], help="Number of major vessels (0-3) colored by fluoroscopy")
    thal = st.selectbox('Thalassemia (thal)', options=[
        (1, 'Normal (1)'), (2, 'Fixed Defect (2)'), (3, 'Reversible Defect (3)')
    ], format_func=lambda x: x[1], help="A blood disorder called Thalassemia")[0]

# --- Prediction Logic ---
st.markdown("---")

if st.button('Analyze Patient Risk', use_container_width=True, type="primary"):
    
    # 1. Create a DataFrame from user inputs (must match model feature order)
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalachh, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # 2. Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # 3. Make prediction
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    # 4. Display Result
    if prediction == 1:
        risk_level = "High Risk - Heart Disease Predicted"
        color = "#B22222" # Red
        message = f"**{risk_level}** (Target=1). The patient has a high probability of heart disease."
        proba_message = f"Confidence (Disease): **{prediction_proba[1]*100:.2f}%**"
    else:
        risk_level = "Low Risk - No Heart Disease Predicted"
        color = "#4CAF50" # Green
        message = f"**{risk_level}** (Target=0). The patient is predicted to be free from heart disease."
        proba_message = f"Confidence (No Disease): **{prediction_proba[0]*100:.2f}%**"

    st.markdown(f"""
        <div class="result-box" style="background-color: #ffffff; border: 3px solid {color};">
            <h3 style="color: {color}; margin-bottom: 10px;">{risk_level}</h3>
            <p style="font-size: 18px;">{message}</p>
            <p style="font-size: 16px; font-style: italic; color: #555;">{proba_message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- DISPLAY PERSONALIZED ADVICE ONLY IF HIGH RISK ---
    if prediction == 1:
        st.markdown("---")
        display_advice(input_data) # Pass the patient data here!


st.markdown("---")
st.caption("Disclaimer: This ML prediction tool is for informational and educational purposes only. Always consult a qualified healthcare professional for medical advice.")
