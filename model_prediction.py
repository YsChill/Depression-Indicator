import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import joblib

# Step 1: Load dataset from CSV file
# This reads the student depression dataset into a pandas DataFrame (a table-like structure in Python)
df = pd.read_csv('Student Depression Dataset.csv')

# Step 2: Remove unnecessary columns
# 'id' and 'City' are not useful for our model, so we remove them
df.drop(columns=['id', 'City'], inplace=True)

# Step 3: Handle missing values
# If 'Financial Stress' has any empty values, we fill them with the median (middle) value of that column
df['Financial Stress'].fillna(df['Financial Stress'].median(), inplace=True)

# Step 4: Encode categorical variables (convert words to numbers)
# Machine learning models cannot process words directly, so we convert categorical data into numerical values.
# We do this using **One-Hot Encoding**, which creates separate binary (0/1) columns for each category.

label_encoders = {}  # This will store the encoders used for later reference
categorical_columns = ['Gender', 'Profession', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Degree']

# Initialize OneHotEncoder (we do not drop any category to keep all information)
ohe_general = OneHotEncoder(sparse_output=False, drop=None)

# Fit the encoder to the categorical columns and transform the data
encoded_general = ohe_general.fit_transform(df[categorical_columns])

# Convert the encoded numpy array into a DataFrame
ohe_general_df = pd.DataFrame(encoded_general, columns=ohe_general.get_feature_names_out(categorical_columns))

# Drop original categorical columns and replace them with encoded versions
df = df.drop(columns=categorical_columns).join(ohe_general_df)

# Save the encoder so we can apply the same transformation to future data
general_encoder_path = 'general_ohe.pkl'
joblib.dump(ohe_general, general_encoder_path)

# Step 5: Scale numerical features
# We want all numbers to be between 0 and 1 to make the model work better
scaler = MinMaxScaler(feature_range=(0, 1))
numerical_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                     'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']

# Apply scaling to numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Round values to 4 decimal places for neatness
df[numerical_columns] = df[numerical_columns].round(4)

# Step 6: Save processed data
df.to_csv('Processed_Student_Depression_Dataset.csv', index=False)

# Step 7: Save the scaler for future use
joblib.dump(scaler, 'minmax_scaler.pkl')

# Step 8: Save conversion descriptions
# This file will contain details on how categorical values were transformed and how numerical values were scaled
encoding_mappings = {}
for col, categories in zip(categorical_columns, ohe_general.categories_):
    encoding_mappings[col] = {category: f"One-Hot Column: {col}_{category}" for category in categories}

scaling_mappings = []
for i, feature in enumerate(numerical_columns):
    original_min = scaler.data_min_[i]
    original_max = scaler.data_max_[i]
    scaling_mappings.append({
        "Feature": feature,
        "Original Min": original_min,
        "Original Max": original_max,
        "Scaled Min": 0,
        "Scaled Max": 1,
        "Formula": f"(value - {original_min}) / ({original_max} - {original_min})"
    })

encoding_df = pd.DataFrame.from_dict(encoding_mappings, orient='index').stack().reset_index()
encoding_df.columns = ['Category', 'Original Value', 'New Value']
scaling_df = pd.DataFrame(scaling_mappings)

# Merge both categorical and numerical mappings into one file
conversion_descriptions = pd.concat([encoding_df, scaling_df], ignore_index=True)
conversion_descriptions.to_csv('Conversion_Descriptions.csv', index=False)

# Step 9: Print confirmation messages
print("Preprocessing complete. Saved dataset:")
print("- 'Processed_Student_Depression_Dataset.csv' (All One-Hot Encoded Categories)")
print("- One-Hot Encoder saved as 'general_ohe.pkl'")
print("- MinMax Scaler saved as 'minmax_scaler.pkl'")
print("- Conversion details saved as 'Conversion_Descriptions.csv'")
