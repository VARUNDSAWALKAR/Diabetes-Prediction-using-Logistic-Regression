# Diabetes Prediction using Logistic Regression

A beginner-friendly machine learning project that predicts whether a patient has diabetes from diagnostic measurements, using **Logistic Regression** on the Pima Indians Diabetes Database.

> **Disclaimer:** This is an educational project. It is not a medical diagnostic tool.

## Objective
Predict `Outcome` (0 = No diabetes, 1 = Diabetes) from eight clinical features:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

## Dataset
- **Pima Indians Diabetes Database** (768 records, 9 columns), originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
- Available on Kaggle (file: `diabetes.csv`). Licence: CC0 / Public Domain.
- The CSV is not committed here. Download it with `python get_data.py` or manually from Kaggle and place it in the project root.

## Methodology
1. Load and inspect the data (shape, types, summary statistics)
2. Replace invalid zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI with `NaN`
3. Stratified 80/20 train-test split (`random_state=42`)
4. Median imputation (fitted on training data only)
5. Feature scaling with `StandardScaler`
6. Logistic Regression with `class_weight='balanced'`
7. Evaluation and coefficient interpretation

## Results
Replace these with the values printed by your notebook run.

| Metric | Score |
| --- | --- |
| Accuracy | _fill in_ |
| Precision | _fill in_ |
| Recall | _fill in_ |
| F1-score | _fill in_ |
| ROC-AUC | _fill in_ |

Visualisations: confusion matrix heatmap, ROC curve, and a feature-importance chart of model coefficients (Glucose and BMI are the strongest positive drivers).

## Why Recall?
In screening, a missed diabetic case (false negative) is costlier than a false alarm (false positive), so recall is prioritised over raw accuracy.

## Limitations
- Small sample (768 rows)
- Specific population (Pima Indian women aged 21+), so results may not generalise
- Missing values in Insulin and SkinThickness are only approximated by median imputation
- Educational benchmark, not validated for clinical use

## How to Run
```bash
git clone https://github.com/<your-username>/diabetes-prediction-logistic-regression.git
cd diabetes-prediction-logistic-regression
pip install -r requirements.txt
python get_data.py          # downloads diabetes.csv
jupyter notebook diabetes_logistic_regression.ipynb
```

## Project Structure
```
.
├── diabetes_logistic_regression.ipynb   # main notebook
├── diabetes_logistic_regression.py      # same code as a script
├── get_data.py                          # downloads the dataset
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Author
Your Name, [GitHub profile](https://github.com/<your-username>)
