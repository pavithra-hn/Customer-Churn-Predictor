import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the Telco Customer Churn dataset"""
    # You can download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    # For now, creating sample data structure - replace with actual data loading
    
    try:
        # Try to load actual dataset
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except:
        # Create sample data if file not found
        print("Dataset not found. Creating sample data for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'customerID': [f'ID_{i}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.round(np.random.uniform(18.0, 120.0, n_samples), 2),
            'TotalCharges': np.round(np.random.uniform(18.0, 8000.0, n_samples), 2),
        })
        
        # Create realistic churn based on some logic
        churn_prob = (
            (df['Contract'] == 'Month-to-month') * 0.4 +
            (df['tenure'] < 12) * 0.3 +
            (df['MonthlyCharges'] > 80) * 0.2 +
            np.random.random(n_samples) * 0.3
        )
        df['Churn'] = (churn_prob > 0.5).astype(int).map({1: 'Yes', 0: 'No'})
    
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    
    return df

def preprocess_features(df):
    """Preprocess features for model training"""
    # Create a copy
    data = df.copy()
    
    # Handle TotalCharges if it's object type
    if data['TotalCharges'].dtype == 'object':
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(0, inplace=True)
    
    # Select features for modeling
    feature_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    X = data[feature_cols].copy()
    y = data['Churn'].copy()
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    return X, y, label_encoders, target_encoder, feature_cols

def train_models(X, y):
    """Train both Logistic Regression and Random Forest models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Train Random Forest (as alternative to XGBoost for simplicity)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Choose best model
    if lr_accuracy >= rf_accuracy:
        best_model = lr_model
        best_accuracy = lr_accuracy
        model_type = 'logistic_regression'
        X_processed = X_train_scaled
        print("Selected: Logistic Regression")
    else:
        best_model = rf_model
        best_accuracy = rf_accuracy
        model_type = 'random_forest'
        X_processed = X_train
        scaler = None
        print("Selected: Random Forest")
    
    return best_model, scaler, best_accuracy, model_type

def main():
    """Main function to train and save model"""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Preprocessing features...")
    X, y, label_encoders, target_encoder, feature_cols = preprocess_features(df)
    
    print("Training models...")
    model, scaler, accuracy, model_type = train_models(X, y)
    
    # Save all artifacts
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_cols': feature_cols,
        'model_type': model_type,
        'accuracy': accuracy
    }
    
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"\nModel saved successfully!")
    print(f"Model Type: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Features: {len(feature_cols)}")

if __name__ == "__main__":
    main()