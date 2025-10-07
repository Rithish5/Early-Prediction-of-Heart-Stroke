Early Heart Stroke Prediction App 🩺: Deployed End-to-End ML Solution
A public-facing data science project demonstrating a complete machine learning pipeline for real-time cardiovascular risk assessment via a Streamlit web application.

🎯 Project Overview
Goal
The primary goal was to build a robust, easily accessible tool to provide early, data-driven insights into heart disease risk, encouraging users to consult medical professionals promptly.

Technical Achievements
Successfully engineered and deployed a Random Forest Classifier model trained on combined clinical datasets. The final application offers actionable insights beyond a simple binary prediction.

Key Deliverables
✅ Quantifiable Risk: Provides a clear percentage probability of heart disease (e.g., 92% Risk Score).

✅ Personalized Guidance: Delivers context-specific health advice (dietary, lifestyle) tailored to the patient's individual high-risk factor inputs (e.g., high cholesterol, elevated BP).

✅ Analytical Dashboard: Integrated data visualizations, including feature distributions, to contextualize patient input against the entire dataset.

✅ Production Ready: Complete ML workflow, from data preprocessing to serialization and public cloud deployment on Streamlit.

🚀 Live Application & Access
The application is deployed and available globally via Streamlit Cloud.

Access the App Here: https://early-prediction-of-heart-stroke-rithishss27032005.streamlit.app/

📁 Project Structure
Heart_Stroke_Prediction_App/
├── app.py                             # Core Streamlit application (UI, prediction, and visualization logic)
├── train_model_final.py               # ML script for data handling, training, and asset serialization
├── requirements.txt                   # Dependency manifest for environment setup
├── model.pkl                          # Final serialized Random Forest Classifier
└── scaler.pkl                         # Final serialized StandardScaler object for feature normalization

📊 Technical Deep Dive: Analysis & Model
Dataset & Modeling Approach
Dataset: Compiled and cleaned clinical records sourced from multiple public heart disease repositories (400+ entries, 13 features).

Target: Binary classification (1 = Risk of heart disease, 0 = No risk).

Methodology: Supervised learning using the Random Forest algorithm, chosen for its robustness and strong interpretability.

Top Predictors Identified by the Model
The model consistently ranked the following features as the most decisive factors in predicting risk:

Chest Pain Type (cp): Differentiating between asymptomatic and various forms of angina.

Thalassemia (thal): The presence and type of this blood disorder.

Maximum Heart Rate Achieved (thalachh): Peak heart rate during exercise.

Features Demonstrated in the Application
Data Dashboard: The dashboard visualizes the differences in key clinical factor means (chol, trestbps) between the high-risk and low-risk patient segments within the training data.

💻 Getting Started (Local Setup)
Prerequisites
Python 3.7+

Serialized assets (model.pkl, scaler.pkl) must be available in the root directory.

Installation
Clone the repository:

git clone <repository-url>
cd Early-Prediction-of-Heart-Stroke

Install dependencies:

pip install -r requirements.txt

Run the Application
streamlit run app.py
# Access the application in your browser at http://localhost:8501
