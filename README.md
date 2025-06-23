# 📱 Customer Churn Predictor

A machine learning web application that predicts whether telecom customers will churn based on their profile and usage patterns.

## 🎯 Features

- **Smart Prediction**: Uses Logistic Regression/Random Forest to predict customer churn
- **Interactive UI**: User-friendly Streamlit interface with form inputs
- **Confidence Scoring**: Shows prediction confidence percentage
- **Feature Importance**: Explains which factors contribute most to the prediction
- **Risk Analysis**: Identifies specific risk factors for each customer

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/customer-churn-predictor.git
cd customer-churn-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model
```bash
python model_training.py
```

### 4. Run Application
```bash
streamlit run app.py
```

## 📊 Dataset

The application uses the Telco Customer Churn dataset with features including:
- Customer demographics (gender, age, partner status)
- Account information (tenure, contract type, payment method)
- Services used (phone, internet, streaming services)
- Charges (monthly and total charges)

## 🔮 Sample Predictions

### High Risk Customer Example:
**Input:**
- Gender: Female
- Senior Citizen: No
- Partner: No
- Dependents: No
- Tenure: 3 months
- Contract: Month-to-month
- Payment Method: Electronic check
- Monthly Charges: $85.00
- Internet Service: Fiber optic
- Additional Services: Minimal

**Output:**
- **Prediction**: WILL CHURN 🚨
- **Confidence**: 78%
- **Top Factor**: Contract type (Month-to-month)
- **Risk Factors**: Short tenure, high charges, electronic check payment

### Low Risk Customer Example:
**Input:**
- Gender: Male
- Senior Citizen: No
- Partner: Yes
- Dependents: Yes
- Tenure: 36 months
- Contract: Two year
- Payment Method: Credit card (automatic)
- Monthly Charges: $65.00
- Internet Service: DSL
- Additional Services: Multiple (Online Security, Tech Support, etc.)

**Output:**
- **Prediction**: WILL NOT CHURN ✅
- **Confidence**: 89%
- **Top Factor**: Contract type (Two year)
- **Risk Analysis**: Low risk profile with good retention indicators

## 🛠️ Model Performance

- **Algorithm**: Logistic Regression / Random Forest (auto-selected based on performance)
- **Accuracy**: ~85% (varies based on data split)
- **Features**: 19 customer attributes
- **Training**: Stratified split with preprocessing

## 📁 Project Structure

```
customer-churn-predictor/
├── app.py                 # Streamlit web application
├── model_training.py      # Model training script
├── requirements.txt       # Python dependencies
├── churn_model.pkl       # Trained model (generated)
├── README.md             # This file
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (optional)
```

## 🌐 Deployment

### Streamlit Cloud Deployment:
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and branch
5. Set main file path: `app.py`
6. Deploy!

### Local Development:
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (creates churn_model.pkl)
python model_training.py

# Run app
streamlit run app.py
```

## 🎯 Usage Instructions

1. **Fill Customer Information**: Enter customer demographics, account details, and service usage
2. **Submit Prediction**: Click "Predict Churn" to get results
3. **Review Results**: 
   - See churn prediction (Will Churn / Will Not Churn)
   - Check confidence percentage
   - Review top contributing factors
   - Analyze risk factors and recommendations

## 🔧 Customization

### Adding New Features:
1. Update `model_training.py` to include new features
2. Retrain model: `python model_training.py`
3. Update form inputs in `app.py`
4. Test with new feature combinations

### Model Tuning:
- Modify algorithms in `train_models()` function
- Adjust hyperparameters for better performance
- Add cross-validation for robust evaluation

## 📈 Model Insights

### Key Churn Indicators:
- **Contract Type**: Month-to-month contracts have highest churn
- **Tenure**: Customers with < 12 months tenure are high risk
- **Payment Method**: Electronic check users churn more
- **Monthly Charges**: High charges (>$75) increase churn risk
- **Services**: Lack of additional services indicates higher risk

### Business Recommendations:
1. **Retention Strategy**: Focus on month-to-month customers
2. **Onboarding**: Improve first-year customer experience
3. **Payment Incentives**: Encourage automatic payment methods
4. **Service Bundling**: Promote additional services to increase stickiness
5. **Pricing**: Review pricing strategy for high-charge segments

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Ensure `churn_model.pkl` exists (run `model_training.py` if missing)
3. Verify Python version compatibility (3.8+)
4. Open an issue on GitHub with error details

---

**Live Demo**: [Your Streamlit Cloud URL]
**Video Demo**: [Your demo video URL]