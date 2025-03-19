# Student Depression Dataset Preprocessing

This repository contains a Python script that preprocesses the **Student Depression Dataset** for machine learning models. It performs data cleaning, categorical encoding, numerical scaling, and saves transformation details for reference.

## 📌 Features
- ✅ **Removes unnecessary columns** (`id`, `City`)
- ✅ **Handles missing values** (`Financial Stress` filled with median)
- ✅ **Encodes categorical features** using One-Hot Encoding
- ✅ **Scales numerical values** between 0 and 1 using Min-Max Scaling
- ✅ **Saves transformation details** in `Conversion_Descriptions.csv`
- ✅ **Exports processed dataset** for model training

## 📂 Files
- 📜 `model_prediction.py` - The main preprocessing script
- 📊 `Processed_Student_Depression_Dataset.csv` - Fully preprocessed dataset
- 🔧 `general_ohe.pkl` - One-Hot Encoder saved for reuse
- 🔧 `minmax_scaler.pkl` - Min-Max Scaler saved for reuse
- 📄 `Conversion_Descriptions.csv` - File detailing categorical and numerical transformations

## 🛠 Installation
1. **Clone this repository:**
   ```sh
   git clone https://github.com/yourusername/student-depression-preprocessing.git
   cd student-depression-preprocessing
   ```
2. **Install dependencies:**
   ```sh
   pip install pandas scikit-learn joblib
   ```

## 🚀 Usage
Run the preprocessing script to clean and transform the dataset:
```sh
python model_prediction.py
```
This will generate a processed dataset and save encoding/scaling details for future reference.

## 🔍 Understanding Transformations
### **Categorical Encoding:**
One-Hot Encoding is applied to categorical columns.

| Category  | Original Value | Encoded Column |
|-----------|---------------|----------------|
| Gender    | Male          | `Gender_Male`  |
| Gender    | Female        | `Gender_Female` |

### **Numerical Scaling:**
Min-Max Scaling transforms numerical values between 0 and 1.

| Feature           | Original Min | Original Max | Scaled Min | Scaled Max | Formula |
|------------------|-------------|-------------|-----------|-----------|---------|
| Age             | 18          | 60          | 0         | 1         | `(value - 18) / (60 - 18)` |
| CGPA            | 2.0         | 4.0         | 0         | 1         | `(value - 2.0) / (4.0 - 2.0)` |

