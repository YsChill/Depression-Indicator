from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and any encoders/scalers
model = joblib.load('../Trained_Model/logistic_regression_model.pkl')  # or 'voting_classifier_model.pkl'
scaler = joblib.load('../Preprocessing/minmax_scaler.pkl')
encoder = joblib.load('../Preprocessing/ohe_general.pkl')

# Feature columns (in same order as training)
numerical_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                     'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
sleep_mapping = {
    'Less than 5 hours': 0,
    '5-6 hours': 0.33,
    '7-8 hours': 0.66,
    'More than 8 hours': 1
}
diet_mapping = {'Unhealthy': 0, 'Moderate': 0.5, 'Healthy': 1}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Prepare input DataFrame
    df = pd.DataFrame([data])

    # Map categorical single features
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_mapping)
    df['Dietary Habits'] = df['Dietary Habits'].map(diet_mapping)

    # One-hot encode other categoricals
    categorical_cols = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Degree']
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Scale numerical
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Combine features
    df = df.drop(columns=categorical_cols)
    input_data = pd.concat([df, encoded_df], axis=1)

    # Force column structure to match training time
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns

    # Drop extra columns that the model wasn't trained on
    input_data = input_data[expected_features]

    # Predict
    pred = model.predict(input_data)[0]
    result = "Likely Depressed" if pred == 1 else "Not Likely Depressed"
    return jsonify({"prediction": int(pred), "label": result})

if __name__ == '__main__':
    app.run(debug=True)
