import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

# --- CONFIGURATION ---
DATA_FILE = 'raw_merged_heart_dataset.csv'
MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'

def load_and_clean_data(file_path):
    """
    Loads the dataset, handles the non-numeric '?' values,
    and imputes the resulting NaNs.
    """
    # Load data, explicitly treating '?' as NaN to resolve the ValueError
    df = pd.read_csv(file_path, na_values=['?'])
    
    print("Dataset loaded and '?' values converted to NaN.")
    
    # Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Impute Missing Values in X
    # We use 'most_frequent' since some missing columns (like ca, thal) are quasi-categorical/discrete
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    # Fit the imputer on X and transform it
    X_imputed = imputer.fit_transform(X)
    
    # Convert the imputed NumPy array back to a Pandas DataFrame
    X_cleaned = pd.DataFrame(X_imputed, columns=X.columns)
    
    print(f"Data cleaned successfully. Shape: {X_cleaned.shape}")
    
    return X_cleaned, y

def train_and_save_model(X, y):
    """Splits data, trains, evaluates, and saves the model and scaler."""
    
    # Split the cleaned data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training: Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", class_report)

    # Save Model and Scaler using pickle
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Model and Scaler Saved to {MODEL_FILE} and {SCALER_FILE}.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        X_clean, y = load_and_clean_data(DATA_FILE)
        train_and_save_model(X_clean, y)
        
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE}' was not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")