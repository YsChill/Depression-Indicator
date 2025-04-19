import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
import os

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'Student Depression Dataset.csv')
df = pd.read_csv(dataset_path)

# Drop unnecessary columns
df.drop(columns=['id', 'City', 'Profession'], inplace=True)

# Fill any remaining NaNs across the dataset with column median
df.fillna(df.median(numeric_only=True), inplace=True)

# One-Hot Encode categorical columns
categorical_columns = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Degree']
ohe_general = OneHotEncoder(sparse_output=False, drop=None)
encoded_general = ohe_general.fit_transform(df[categorical_columns])
ohe_general_df = pd.DataFrame(encoded_general, columns=ohe_general.get_feature_names_out(categorical_columns))
df = df.drop(columns=categorical_columns).join(ohe_general_df)

# Save the encoder
general_encoder_path = os.path.join(script_dir, 'ohe_general.pkl')
joblib.dump(ohe_general, general_encoder_path)

# Map custom numerical features
sleep_mapping = {
    'Less than 5 hours': 0,
    '5-6 hours': 0.33,
    '7-8 hours': 0.66,
    'More than 8 hours': 1
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_mapping)

dietary_mapping = {
    'Unhealthy': 0,
    'Moderate': 0.5,
    'Healthy': 1
}
df['Dietary Habits'] = df['Dietary Habits'].map(dietary_mapping)

# Scale numerical features
scaler = MinMaxScaler(feature_range=(0, 1))
numerical_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                     'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Round scaled values and custom maps
df[numerical_columns + ['Sleep Duration', 'Dietary Habits']] = df[numerical_columns + ['Sleep Duration', 'Dietary Habits']].round(4)


# Save final processed dataset
processed_data_path = os.path.join(script_dir, 'Processed_Student_Depression_Dataset.csv')
df.to_csv(processed_data_path, index=False)

# Save the scaler
scaler_path = os.path.join(script_dir, 'minmax_scaler.pkl')
joblib.dump(scaler, scaler_path)

# Save transformation descriptions
encoding_mappings = {
    col: {cat: f"{col}_{cat}" for cat in cats}
    for col, cats in zip(categorical_columns, ohe_general.categories_)
}
scaling_mappings = [{
    "Feature": col,
    "Original Min": scaler.data_min_[i],
    "Original Max": scaler.data_max_[i],
    "Scaled Min": 0,
    "Scaled Max": 1,
    "Formula": f"(value - {scaler.data_min_[i]}) / ({scaler.data_max_[i]} - {scaler.data_min_[i]})"
} for i, col in enumerate(numerical_columns)]

scaling_mappings.append({
    "Feature": "Sleep Duration",
    "Original Values": sleep_mapping,
    "Scaled Values": {v: round(v, 2) for v in sleep_mapping.values()}
})
scaling_mappings.append({
    "Feature": "Dietary Habits",
    "Original Values": dietary_mapping,
    "Scaled Values": {v: round(v, 2) for v in dietary_mapping.values()}
})

encoding_df = pd.DataFrame.from_dict(encoding_mappings, orient='index').stack().reset_index()
encoding_df.columns = ['Category', 'Original Value', 'New Value']
scaling_df = pd.DataFrame(scaling_mappings)
conversion_descriptions = pd.concat([encoding_df, scaling_df], ignore_index=True)
conversion_descriptions_path = os.path.join(script_dir, 'Conversion_Descriptions.csv')
conversion_descriptions.to_csv(conversion_descriptions_path, index=False)

# Final message
print("Preprocessing complete. Files saved:")
print("- Processed_Student_Depression_Dataset.csv")
print("- ohe_general.pkl")
print("- minmax_scaler.pkl")
print("- Conversion_Descriptions.csv")
