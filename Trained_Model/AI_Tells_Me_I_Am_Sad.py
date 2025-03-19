import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed dataset
df = pd.read_csv('Processed_Student_Depression_Dataset.csv')

# Define features (X) and target variable (y)
y = df['Depression']  # Assuming 'Depression' is the target column
X = df.drop(columns=['Depression'])

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'depression_prediction_model.pkl')
print("Model saved as 'depression_prediction_model.pkl'")
