import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'
DATA_FILE = 'raw_merged_heart_dataset.csv' # Added data file for visualization

# Set the page title/icon
# We set the layout to 'wide' for the dashboard to have more space
st.set_page_config(page_title="Early Heart Stroke Prediction App", layout="wide")

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


# --- Load Model, Scaler, and Data ---
model = None
scaler = None
df_full = None # Variable to hold the entire dataset for visualization

try:
    # Load Model and Scaler
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        st.error("Error: Model files ('model.pkl' and 'scaler.pkl') not found.")
        st.warning("Please run trained_model.py to generate these files.")
        st.stop()
        
    with open(MODEL_FILE, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_FILE, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load Full Dataset for Visualization
    if os.path.exists(DATA_FILE):
        df_raw = pd.read_csv(DATA_FILE, na_values=['?'])
        
        # Simple imputation logic (assuming no separate imputer file exists, 
        # so we perform a basic imputation for plotting)
        imputer = None
        if os.path.exists('imputer.pkl'):
             imputer = pickle.load(open('imputer.pkl', 'rb'))
        
        from sklearn.impute import SimpleImputer
        if imputer is None:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            
        feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        df_features = df_raw[feature_cols].copy()
        
        # Fit and transform (if not fit before)
        try:
            df_imputed = imputer.fit_transform(df_features)
        except:
             # If fit fails, try transforming based on existing fit (safer for deployment)
             df_imputed = imputer.transform(df_features)

        df_full = pd.DataFrame(df_imputed, columns=feature_cols)
        # Ensure target is binary (0 for Low Risk, 1 for High Risk)
        df_full['target'] = df_raw['target'].apply(lambda x: 1 if x > 0 else 0)
        
    else:
        st.warning(f"Data file '{DATA_FILE}' not found. Data Visualization will be limited.")
        
except Exception as e:
    st.error(f"An error occurred during file loading: {e}")
    st.stop()


# --- Custom Personalized Advice Function ---
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


# --- Dashboard Visualization Function (New and Improved) ---
def display_data_dashboard(age, df):
    """Shows multiple charts for data exploration."""
    st.markdown("## 📊 Data Analytics Dashboard")
    st.info("Explore the dataset's characteristics to understand the risk factors contextually.")
    
    # Ensure 'age' is numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    col_dash1, col_dash2 = st.columns(2)
    
    # --- Chart 1: Age Distribution Histogram (from previous update) ---
    with col_dash1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x='age', hue='target', kde=True, bins=20, palette={0: '#0077b6', 1: '#B22222'}, ax=ax1)
        ax1.axvline(age, color='black', linestyle='--', linewidth=2, label=f"Patient Age: {age}")
        ax1.set_title('Age Distribution of Dataset (Risk vs. No Risk)')
        ax1.set_xlabel('Age (Years)')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)

    # --- Chart 2: Risk Factor Means by Outcome ---
    with col_dash2:
        risk_cols = ['chol', 'trestbps', 'thalachh']
        df_mean = df.groupby('target')[risk_cols].mean().T
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        df_mean.plot(kind='bar', ax=ax2, color=['#0077b6', '#B22222'])
        ax2.set_title('Average Key Factors by Outcome')
        ax2.set_xlabel('Risk Factor')
        ax2.set_ylabel('Average Value')
        ax2.legend(['Low Risk (0)', 'High Risk (1)'])
        plt.xticks(rotation=0)
        st.pyplot(fig2)

    # --- Chart 3: Gender and Chest Pain (cp) Distribution ---
    st.markdown("### Categorical Feature Analysis")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    
    # Map numerical categories to labels for better readability on the chart
    df['cp_label'] = df['cp'].map({0: 'Asymptomatic', 1: 'Typical Angina', 
                                   2: 'Atypical Angina', 3: 'Non-Anginal'})
    df['sex_label'] = df['sex'].map({1: 'Male', 0: 'Female'})

    # Create the count plot
    sns.countplot(x='cp_label', hue='target', data=df, palette={0: '#0077b6', 1: '#B22222'}, ax=ax3)
    
    ax3.set_title('Chest Pain Type vs. Risk Outcome')
    ax3.set_xlabel('Chest Pain Type (cp)')
    ax3.set_ylabel('Count')
    ax3.legend(title='Outcome', labels=['Low Risk (0)', 'High Risk (1)'])
    plt.xticks(rotation=15)
    st.pyplot(fig3)


# --- Streamlit UI Setup (Reverted to Original Colors) ---

st.markdown("""
    <style>
    /* Ensured default Streamlit colors are used but kept the result box styling */
    .big-font {
        font-size:32px !important;
        font-weight: bold;
        color: #0077b6; /* Streamlit Primary Color (Default Blue) */
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
    sex_input = st.selectbox('Sex', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1])
    sex = sex_input[0]
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

    # --- DISPLAY VISUALIZATION ---
    st.markdown("---")
    if df_full is not None:
        display_data_dashboard(age, df_full.copy())
    else:
        st.warning("Cannot display full dashboard: Data file ('raw_merged_heart_dataset.csv') not found.")


st.markdown("---")
st.caption("Disclaimer: This ML prediction tool is for informational and educational purposes only. Always consult a qualified healthcare professional for medical advice.")
