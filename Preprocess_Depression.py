import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv('Student Depression Dataset.csv')

# Drop irrelevant columns
df.drop(columns=['id', 'City'], inplace=True)

# Handle missing values
df['Financial Stress'].fillna(df['Financial Stress'].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree',
                       'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                     'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save preprocessed dataset
df.to_csv('Processed_Student_Depression_Dataset.csv', index=False)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

print("Preprocessing complete. Saved as 'Processed_Student_Depression_Dataset.csv' and scaler saved as 'scaler.pkl'.")
