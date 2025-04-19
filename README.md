
# 🧠 Depression Indicator - Machine Learning Pipeline

This project predicts signs of **student depression** using a machine learning pipeline built with Python. It covers data preprocessing, model training, prediction through a REST API, and a simple GUI for user interaction.

## 🚦 Project Components

- 🧹 **Preprocessing Script** (`model_prediction.py`)  
  Cleans, encodes, and scales raw data, then exports a ready-to-train dataset.
  
- 🧠 **Model Training** (`AI_Tells_Me_I_Am_Sad.py`)  
  Trains multiple classifiers and saves the best-performing models (e.g., Logistic Regression, XGBoost, Voting Classifier).

- 🌐 **API Server** (`app.py`)  
  Flask-based API that loads the trained model and accepts JSON input to return predictions.

- 💻 **GUI Interface** (`gui.py`)  
  Tkinter desktop app for entering student data and viewing depression prediction results.

## 📦 Files & Outputs

- `Student Depression Dataset.csv` - Raw dataset
- `Processed_Student_Depression_Dataset.csv` - Cleaned and encoded dataset
- `Conversion_Descriptions.csv` - Encoding & scaling details
- `minmax_scaler.pkl` - Saved MinMaxScaler
- `ohe_general.pkl` - OneHotEncoder for categorical features
- `*_model.pkl` - Trained models (e.g. `logistic_regression_model.pkl`, `voting_classifier_model.pkl`)
- `app.py` - Flask API to serve model predictions
- `gui.py` - Tkinter frontend for interactive input

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YsChill/Depression-Indicator.git
   cd Depression-Indicator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually:
   ```bash
   pip install pandas scikit-learn flask joblib xgboost lightgbm
   ```

3. **Preprocess the dataset**
   ```bash
   python model_prediction.py
   ```

4. **Train the models**
   ```bash
   python AI_Tells_Me_I_Am_Sad.py
   ```

5. **Run the Flask API**
   ```bash
   python app.py
   ```

6. **Launch the GUI**
   ```bash
   python gui.py
   ```

## ⚠️ Note on Model & Preprocessing Files

This project ignores large `.pkl` files (model, encoder, scaler) to keep the repository lightweight.

To generate these files locally:

1. Run the preprocessing script:
   ```bash
   python model_prediction.py
   ```

2. Train the models:
   ```bash
   python AI_Tells_Me_I_Am_Sad.py
   ```

## 🔍 Sample Transformations

### One-Hot Encoding (Categorical)

| Column                      | Original Value | Encoded Column            |
|----------------------------|----------------|----------------------------|
| Gender                     | Male           | Gender_Male                |
| Family History of Mental Illness | Yes      | Family History of Mental Illness_Yes |
| Degree                     | B.Tech         | Degree_B.Tech              |

### Min-Max Scaling (Numerical)

| Feature           | Original Range | Scaled Range | Formula                        |
|------------------|----------------|--------------|--------------------------------|
| Age              | 18 - 60        | 0 - 1        | `(value - 18) / (60 - 18)`     |
| CGPA             | 2.0 - 4.0      | 0 - 1        | `(value - 2.0) / (4.0 - 2.0)`  |

## ✨ Live Demo (Coming Soon)
Future versions will include a web-based frontend and database integration.
