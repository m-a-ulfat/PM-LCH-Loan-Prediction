from flask import Flask, render_template, request
from catboost import CatBoostClassifier
import pandas as pd

app = Flask(__name__)

# Load CatBoost model once when the app starts
model = CatBoostClassifier()
model.load_model("PMLP_catboost_model7.1.1.cbm")

# Store the expected feature names from the model
expected_features = model.feature_names_

# Utility: Safely convert to numeric
def safe_numeric(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# Format the input for prediction
def format_input(form):
    # Raw form input
    raw_data = {
        'Region': form.get('region'),
        'Area': form.get('area'),
        'Branch': form.get('branch'),
        'Branch Code': (form.get('branch_code')),
        'Gender': form.get('gender'),
        'Age': (form.get('age')),
        'Age Level': form.get('age_level'),
        'Education': form.get('education'),
        'Marital Status': form.get('marital_status'),
        'House_ownership': form.get('house_ownership'),
        'Total_family_members': (form.get('total_family_members')),
        'No_of_earning_hands':(form.get('no_of_earning_hands')),
        'Source of Incom': form.get('source_of_income'),
        'Incom': (form.get('income')),
        'Expenses':(form.get('expenses')),
        'Monthly_Saving': form.get('monthly_saving'),
        'Saving Amount': (form.get('saving_amount')),
        'who_will_earn': form.get('who_will_earn'),
        'Social behavior': form.get('social_behavior'),
        'Loan Amount': (form.get('loan_amount')),
        'Inst Months': (form.get('inst_months') or 0),
        'Inst Amnt': (form.get('inst_amnt')),
        'Activity': form.get('activity'),
    }

    # Convert to DataFrame
    df = pd.DataFrame([raw_data])

    # Normalize column names to match model
    # Rename columns to match expected_features exactly
    rename_map = {col: ef for ef in expected_features for col in df.columns if col.lower() == ef.lower()}
    df.rename(columns=rename_map, inplace=True)

    # Reorder columns to match model input order
    df = df[expected_features]

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_df = format_input(request.form)
        prediction = model.predict(input_df)
        result = prediction[0]
        return render_template('index.html', prediction_text=f'Tax Status Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
