import tkinter as tk
from tkinter import ttk
import requests

# Mapping labels to numeric values
scale_map = {
    "Low": 1,
    "Sorta Low": 2,
    "Medium": 3,
    "Sorta High": 4,
    "High": 5
}

sleep_options = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
diet_options = ["Unhealthy", "Moderate", "Healthy"]
gender_options = ["Male", "Female"]
yes_no = ["Yes", "No"]
degree_options = [
    "B.Arch", "B.Com", "B.Ed", "B.Pharm", "B.Tech", "BA", "BBA", "BCA", "BE", "BHM", "BSc",
    "Class 12", "LLB", "LLM", "M.Com", "M.Ed", "M.Pharm", "M.Tech", "MA", "MBA", "MBBS", "MCA",
    "MD", "ME", "MHM", "MSc", "Others", "PhD"
]

def submit():
    try:
        data = {
            "Age": int(age.get()),
            "Academic Pressure": scale_map[academic.get()],
            "Work Pressure": scale_map[work.get()],
            "CGPA": float(cgpa.get()),
            "Study Satisfaction": scale_map[study.get()],
            "Job Satisfaction": scale_map[job.get()],
            "Work/Study Hours": float(hours.get()),
            "Financial Stress": scale_map[financial.get()],
            "Sleep Duration": sleep.get(),
            "Dietary Habits": diet.get(),
            "Gender": gender.get(),
            "Have you ever had suicidal thoughts ?": suicide.get(),
            "Family History of Mental Illness": family.get(),
            "Degree": degree.get()
        }

        response = requests.post("http://localhost:5000/predict", json=data)
        result = response.json()
        output_label.config(text=f"Prediction: {result['label']}")
    except Exception as e:
        output_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
root.title("Depression Prediction")

# Layout definitions
def add_label_dropdown(row, label_text, options):
    tk.Label(root, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=3)
    combo = ttk.Combobox(root, values=options, state="readonly", width=30)
    combo.grid(row=row, column=1, padx=5, pady=3)
    combo.current(0)
    return combo

def add_label_entry(row, label_text):
    tk.Label(root, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=3)
    entry = tk.Entry(root, width=32)
    entry.grid(row=row, column=1, padx=5, pady=3)
    return entry

row = 0
age = add_label_entry(row, "Age (e.g. 18–30):"); row += 1
academic = add_label_dropdown(row, "Academic Pressure:", list(scale_map.keys())); row += 1
work = add_label_dropdown(row, "Work Pressure:", list(scale_map.keys())); row += 1
cgpa = add_label_entry(row, "CGPA (0.0–4.0):"); row += 1
study = add_label_dropdown(row, "Study Satisfaction:", list(scale_map.keys())); row += 1
job = add_label_dropdown(row, "Job Satisfaction:", list(scale_map.keys())); row += 1
hours = add_label_entry(row, "Work/Study Hours (per day):"); row += 1
financial = add_label_dropdown(row, "Financial Stress:", list(scale_map.keys())); row += 1
sleep = add_label_dropdown(row, "Sleep Duration:", sleep_options); row += 1
diet = add_label_dropdown(row, "Dietary Habits:", diet_options); row += 1
gender = add_label_dropdown(row, "Gender:", gender_options); row += 1
suicide = add_label_dropdown(row, "Have you ever had suicidal thoughts?", yes_no); row += 1
family = add_label_dropdown(row, "Family History of Mental Illness:", yes_no); row += 1
degree = add_label_dropdown(row, "Degree:", degree_options); row += 1

# Submit and output
submit_btn = tk.Button(root, text="Predict", command=submit)
submit_btn.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

output_label = tk.Label(root, text="Prediction: ")
output_label.grid(row=row, column=0, columnspan=2)

root.mainloop()
