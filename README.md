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


## INSIGHTS
<img width="484" height="334" alt="image" src="https://github.com/user-attachments/assets/8fcfd322-2663-481f-8b13-714f90a6d046" />

The class distribution plot reveals a clear imbalance in the dataset, containing 500 non-diabetic cases (Class 0) compared to only 268 diabetic cases (Class 1), establishing a majority-to-minority ratio of roughly 1.87 to 1. Under this skew, an unadjusted model could easily bias its predictions toward the negative class to achieve an artificially high accuracy score while failing to detect actual diabetic patients. To counter this, the pipeline incorporates a balanced class weighting strategy (`class_weight='balanced'`) during logistic regression training, which places greater penalty on misclassifying the minority group. Addressing this distribution gap directly supports the project's clinical objective: maximizing recall to ensure as many true positive cases are captured as possible during diagnostic screening.

<img width="509" height="434" alt="image" src="https://github.com/user-attachments/assets/9cf7f481-ef3c-4b6f-8b00-2a80db19d6ff" />
The confusion matrix for the test set demonstrates how the balanced logistic regression model distributes its classification decisions across 154 total test instances. Out of 100 actual non-diabetic patients (Actual 0), the model correctly identifies 75 true negatives ({TN} = 75) while misclassifying 25 as false positives ({FP} = 25). More importantly for clinical evaluation, out of 54 actual diabetic patients (Actual 1), it successfully captures 38 true positives ({TP} = 38), leaving 16 false negatives ({FN} = 16). This breakdown highlights the trade-off inherent in a diagnostic screening setup: while accepting 25 false alarms reduces precision to 60.3% ({38}/{38+25}), it prioritizes medical safety by restricting missed diabetic cases to 16, directly yielding the model's 70.4% recall rate ({38}/{38+16}).\

<img width="584" height="484" alt="image" src="https://github.com/user-attachments/assets/ce7b4c88-3c81-4f66-83d9-62b5cd3bafac" />
The Receiver Operating Characteristic (ROC) curve evaluates the trade-off between the True Positive Rate (Recall) and the False Positive Rate across every potential decision threshold for the test set. The model achieves an Area Under the Curve ({AUC}) of 0.813, substantially outperforming the baseline random guess diagonal ({AUC} = 0.500). This steep climb in the lower False Positive Rate range indicates that the logistic regression model maintains robust discriminatory power, reliably ranking positive diabetes cases higher than non-diabetic cases across varying operational thresholds. In a screening framework, such strong separability confirms that decision thresholds can be flexibly tuned to further minimize false negatives without inducing an excessive surge in false alarms.

<img width="785" height="484" alt="image" src="https://github.com/user-attachments/assets/836cdcc8-e97e-4aa7-b957-1a0135d016dd" />
The feature importance plot illustrates the standardized logistic regression coefficients, quantifying each variable's relative impact on the log-odds of a positive diabetes diagnosis per one standard deviation increase. Plasma glucose concentration emerged as the primary determinant with the highest positive coefficient (+1.18), closely followed by BMI (+0.71), establishing both metabolic indicators as the dominant risk factors driving the classification model. Secondary clinical variables such as number of pregnancies (+0.37), diabetes pedigree function (+0.29), and age (+0.19) demonstrated moderate positive associations with diabetes likelihood. Conversely, skin thickness (+0.01), blood pressure (-0.01), and insulin (-0.04) exhibited near-zero coefficients, indicating minimal independent predictive contribution in the presence of dominant metabolic predictors like glucose and BMI.



This project built a Logistic Regression classifier on the Pima Indians Diabetes dataset to predict
diabetes (Outcome = 1). Invalid zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI were
treated as missing values and filled with the training-set median, features were standardised, and a
stratified 80/20 split (random_state=42) with class_weight='balanced' was used to handle class
imbalance. On the held-out test set the model achieved an accuracy of 0.734, precision of 0.603,
recall of 0.704, an F1-score of 0.650 and a ROC-AUC of 0.813, correctly detecting 38 of 54 diabetic
cases while missing 16. Glucose and BMI had the strongest positive coefficients, in line with their
known clinical role as risk factors. Recall is the key metric for screening because a missed case
(false negative) is costlier than a false alarm. However, the dataset is small and drawn from a
specific population, so the model is suitable for learning classification concepts and not for
medical diagnosis.

## Author
VARUN D SAWALKAR, [GitHub profile](https://github.com/VARUNDSAWALKAR)
