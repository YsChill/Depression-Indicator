import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Load Data
df = pd.read_csv("../Preprocessing/Processed_Student_Depression_Dataset.csv")
df.fillna(df.median(numeric_only=True), inplace=True)
y = df['Depression']
X = df.drop(columns=['Depression'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(),
    "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

# Variables for Later
results = {}
trained_models = {}

# Train Models, Evaluate Them and Save Them
for name, model in models.items():
    print(f"\n🚀 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"📊 Report:\n{classification_report(y_test, y_pred)}")
    joblib.dump(model, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl")
    results[name] = acc
    trained_models[name] = model

# Combine in a VotingClassifier (soft voting where possible)
voting_models = [
    (name, model) for name, model in trained_models.items()
    if name not in ["Neural Net (MLP)", "KNN", "Naive Bayes"]
]

voting_clf = VotingClassifier(estimators=voting_models, voting='soft')
print("\n🤝 Training VotingClassifier...")
voting_clf.fit(X_train, y_train)
y_pred_vote = voting_clf.predict(X_test)
acc_vote = accuracy_score(y_test, y_pred_vote)
print(f"\n🏆 VotingClassifier Accuracy: {acc_vote:.4f}")
print("📊 Report:\n", classification_report(y_test, y_pred_vote))
joblib.dump(voting_clf, "voting_classifier_model.pkl")
print("\n✅ VotingClassifier saved as 'voting_classifier_model.pkl'")

# Print overall comparison
print("\n📈 Model Accuracy Comparison:")
results["VotingClassifier"] = acc_vote
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:25s}: {acc:.4f}")