import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    try:
        with open('churn_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        st.error("Model file not found! Please run model_training.py first.")
        return None

def predict_churn(input_data, artifacts):
    """Make prediction and return results with confidence"""
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col, encoder in artifacts['label_encoders'].items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform([str(input_data[col])])
            except ValueError:
                # Handle unseen categories
                input_df[col] = 0
    
    # Ensure all features are present
    for col in artifacts['feature_cols']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[artifacts['feature_cols']]
    
    # Scale if needed
    if artifacts['scaler'] is not None:
        input_scaled = artifacts['scaler'].transform(input_df)
        prediction = artifacts['model'].predict(input_scaled)
        prediction_proba = artifacts['model'].predict_proba(input_scaled)
    else:
        prediction = artifacts['model'].predict(input_df)
        prediction_proba = artifacts['model'].predict_proba(input_df)
    
    # Get feature importance or coefficients
    if artifacts['model_type'] == 'random_forest':
        feature_importance = artifacts['model'].feature_importances_
    else:
        feature_importance = abs(artifacts['model'].coef_[0])
    
    # Create feature importance dictionary
    importance_dict = dict(zip(artifacts['feature_cols'], feature_importance))
    top_feature = max(importance_dict, key=importance_dict.get)
    
    # Decode prediction
    prediction_label = artifacts['target_encoder'].inverse_transform(prediction)[0]
    confidence = max(prediction_proba[0])
    
    return prediction_label, confidence, top_feature, importance_dict

def main():
    # Title and description
    st.title("📱 Customer Churn Predictor")
    st.markdown("""
    Predict whether a telecom customer will churn based on their profile and usage patterns.
    """)
    
    # Load model
    artifacts = load_model()
    if artifacts is None:
        st.stop()
    
    # Display model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", artifacts['model_type'].replace('_', ' ').title())
    with col2:
        st.metric("Accuracy", f"{artifacts['accuracy']:.2%}")
    with col3:
        st.metric("Features", len(artifacts['feature_cols']))
    
    st.divider()
    
    # Create input form
    st.header("Customer Information")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            
            st.subheader("Account Info")
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            contract = st.selectbox("Contract Type", 
                                  ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", 
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", 
                                        ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", 
                                          ["No", "DSL", "Fiber optic"])
            
            # Internet-dependent services
            if internet_service == "No":
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
            else:
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            
            st.subheader("Charges")
            monthly_charges = st.number_input("Monthly Charges ($)", 
                                            min_value=18.0, max_value=120.0, value=65.0)
            total_charges = st.number_input("Total Charges ($)", 
                                          min_value=18.0, max_value=8000.0, 
                                          value=monthly_charges * tenure)
        
        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Make prediction
            prediction, confidence, top_feature, importance_dict = predict_churn(input_data, artifacts)
            
            st.divider()
            st.header("Prediction Results")
            
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                if prediction == "Yes":
                    st.error(f"🚨 **WILL CHURN**")
                else:
                    st.success(f"✅ **WILL NOT CHURN**")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Explanation
            st.subheader("Key Insights")
            st.info(f"**Top Contributing Factor:** {top_feature.replace('_', ' ').title()}")
            
            # Feature importance chart
            st.subheader("Feature Importance")
            
            # Get top 10 features
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)
            
            fig = px.bar(
                x=list(importances),
                y=[f.replace('_', ' ').title() for f in features],
                orientation='h',
                title="Top 10 Most Important Factors",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors analysis
            st.subheader("Risk Analysis")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract increases churn risk")
            if tenure < 12:
                risk_factors.append("Low tenure (< 12 months) indicates higher risk")
            if monthly_charges > 75:
                risk_factors.append("High monthly charges may lead to churn")
            if internet_service == "Fiber optic" and online_security == "No":
                risk_factors.append("Fiber customers without security are at risk")
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment method shows higher churn")
            
            if risk_factors:
                st.warning("**Risk Factors Identified:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("**Low Risk Profile:** Customer shows good retention indicators")

    # Sidebar with sample inputs
    with st.sidebar:
        st.header("Sample Inputs")
        st.markdown("""
        **High Risk Customer:**
        - Contract: Month-to-month
        - Tenure: 3 months
        - Monthly Charges: $85
        - Payment: Electronic check
        - Internet: Fiber optic
        - No additional services
        
        **Low Risk Customer:**
        - Contract: Two year
        - Tenure: 24 months
        - Monthly Charges: $55
        - Payment: Credit card (automatic)
        - Multiple services: Yes
        """)
        
        st.divider()
        st.markdown("**Model Information:**")
        st.write(f"Trained on telecom customer data")
        st.write(f"Predicts churn probability")
        st.write(f"Accuracy: {artifacts['accuracy']:.1%}")

if __name__ == "__main__":
    main()