# %% [markdown]
# # Diabetes Prediction using Logistic Regression
# **Dataset:** Pima Indians Diabetes Database (`diabetes.csv`, Kaggle)  
# **Target:** `Outcome` (0 = No diabetes, 1 = Diabetes)  
# **Model:** Median imputation → StandardScaler → Logistic Regression (`class_weight='balanced'`)
# 
# **Workflow:** load CSV → inspect → replace invalid zeros with NaN → stratified split → impute → scale → train → evaluate → interpret coefficients.

# %% [markdown]
# ## 1. Imports and Configuration

# %%
import os
import warnings
try:
    from IPython.display import display
except ImportError:          # plain Python fallback
    display = print
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

# ---- Configuration (single place to change settings) ----
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Outcome"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
# Columns where a value of 0 is physiologically impossible (i.e. a missing value)
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Candidate locations of the CSV (local folder, Kaggle, Colab)
DATA_PATHS = ["diabetes.csv", "../input/review/diabetes.csv",
              "/kaggle/input/review/diabetes.csv", "/content/diabetes.csv",
              "/mnt/user-data/uploads/diabetes.csv"]

# %% [markdown]
# ## 2. Load the Data

# %%
def load_data(paths):
    """Load diabetes.csv from the first path that exists."""
    for p in paths:
        if os.path.exists(p):
            print(f"Loaded data from: {p}")
            return pd.read_csv(p)
    # Optional: Kaggle API fallback (uncomment if kagglehub is installed)
    # import kagglehub; d = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    # return pd.read_csv(os.path.join(d, "diabetes.csv"))
    # Google Colab fallback: opens an upload dialog
    try:
        from google.colab import files
        print("diabetes.csv not found - please upload it now...")
        up = files.upload()
        return pd.read_csv(next(iter(up)))
    except ImportError:
        raise FileNotFoundError("diabetes.csv not found. Place it next to this notebook "
                                "or update DATA_PATHS.")

df = load_data(DATA_PATHS)
df.head()

# %% [markdown]
# ## 3. Inspect the Data (shape, types, summary statistics)

# %%
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values (NaN):\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
display(df.describe().T.round(2))

print("\nClass distribution (Outcome):")
print(df[TARGET].value_counts().rename({0: "No diabetes (0)", 1: "Diabetes (1)"}))
print((df[TARGET].value_counts(normalize=True) * 100).round(1).astype(str) + " %")

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
counts = df[TARGET].value_counts().sort_index()
ax.bar(["No diabetes (0)", "Diabetes (1)"], counts.values, color=["#66c2a5", "#fc8d62"])
ax.set_title("Class Distribution")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Replace Invalid Zeros with NaN
# Zero is not a valid value for glucose, blood pressure, skin thickness, insulin or BMI, so these zeros are really *missing* values.

# %%
df_clean = df.copy()
zero_counts = (df_clean[ZERO_INVALID_COLS] == 0).sum()
print("Invalid zeros found per column:\n", zero_counts.to_string())

df_clean[ZERO_INVALID_COLS] = df_clean[ZERO_INVALID_COLS].replace(0, np.nan)

print("\nMissing values after replacement:")
print(pd.DataFrame({"missing": df_clean.isna().sum(),
                    "percent": (df_clean.isna().mean() * 100).round(1)})
      .loc[ZERO_INVALID_COLS])

# %% [markdown]
# ## 5. Stratified Train-Test Split
# The split is done **before** imputation and scaling so that medians and means are learned from the training data only (no data leakage). `stratify=y` keeps the class ratio equal in both sets.

# %%
X = df_clean[FEATURES]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print("Train positive rate:", round(y_train.mean(), 3),
      "| Test positive rate:", round(y_test.mean(), 3))

# %% [markdown]
# ## 6. Build and Train the Pipeline
# Median imputation → StandardScaler → Logistic Regression with `class_weight='balanced'`.

# %%
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced",
                               max_iter=1000,
                               random_state=RANDOM_STATE)),
])

model.fit(X_train, y_train)
print("Model trained successfully.")
print(model)

# %% [markdown]
# ## 7. Evaluate the Model

# %%
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1

metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall":    recall_score(y_test, y_pred),
    "F1-score":  f1_score(y_test, y_pred),
    "ROC-AUC":   roc_auc_score(y_test, y_prob),
}

print("=" * 34)
print(" TEST SET PERFORMANCE")
print("=" * 34)
for name, value in metrics.items():
    print(f"{name:<10}: {value:.3f}")
print("=" * 34)

print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=["No diabetes", "Diabetes"]))

# %% [markdown]
# ### 7.1 Confusion Matrix

# %%
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Annotate each cell with its label and count
labels = np.array([[f"TN = {tn}", f"FP = {fp}"],
                   [f"FN = {fn}", f"TP = {tp}"]])

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
            annot_kws={"size": 13, "weight": "bold"}, ax=ax)
ax.set_title("Confusion Matrix (Test Set)")
plt.tight_layout(); plt.show()

print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Specificity (true negative rate): {tn / (tn + fp):.3f}")

# %% [markdown]
# ### 7.2 ROC Curve

# %%
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = metrics["ROC-AUC"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"Logistic Regression (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier (AUC = 0.500)")
ax.fill_between(fpr, tpr, alpha=0.12, color="darkorange")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve (Test Set)")
ax.legend(loc="lower right")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 7.3 Feature Importance (Model Coefficients)
# Features are standardised, so coefficient sizes are directly comparable. Positive = raises the odds of diabetes.

# %%
coefs = model.named_steps["clf"].coef_[0]
coef_df = (pd.DataFrame({"Feature": FEATURES, "Coefficient": coefs})
           .assign(Odds_Ratio=lambda d: np.exp(d["Coefficient"]))
           .sort_values("Coefficient", ascending=True)
           .reset_index(drop=True))

# Highlight Glucose and BMI as key drivers
KEY_DRIVERS = {"Glucose", "BMI"}
colors = ["crimson" if f in KEY_DRIVERS else
          ("steelblue" if c > 0 else "grey")
          for f, c in zip(coef_df["Feature"], coef_df["Coefficient"])]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="black")
ax.axvline(0, color="black", lw=0.8)
for bar, val in zip(bars, coef_df["Coefficient"]):
    ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=10)
ax.set_xlabel("Standardised coefficient (log-odds per 1 SD)")
ax.set_title("Feature Importance: Logistic Regression Coefficients\n(red = Glucose & BMI, the key drivers)")
plt.tight_layout(); plt.show()

display(coef_df.sort_values("Coefficient", ascending=False).round(3))
top2 = coef_df.sort_values("Coefficient", ascending=False)["Feature"].head(2).tolist()
print("Strongest positive coefficients:", top2)

# %% [markdown]
# ## 8. Project Insights

# %% [markdown]
# ### 8.1 Why Recall is prioritised over raw Accuracy in screening
# 
# * **The two errors are not equally costly.** A *false negative* means a person with diabetes is told they are fine, so the condition goes undetected and untreated and complications can develop. A *false positive* usually only leads to a follow-up blood test.
# * **Accuracy can be misleading with imbalanced classes.** About 65% of patients here are non-diabetic, so a model that always predicts "No diabetes" would score roughly 65% accuracy while detecting **no** cases (recall = 0).
# * **Recall = TP / (TP + FN)** directly measures how many real cases the model catches, so it is the metric screening tools are tuned for. This is why `class_weight='balanced'` is used: it penalises missed positives more heavily, trading some precision for higher recall.
# * Precision and F1 still matter, since too many false alarms overload clinics, so they are reported alongside recall.
# 
# ### 8.2 Dataset Limitations
# 
# * **Small sample:** only 768 records (about 268 positive), so results vary with the split and estimates are noisy.
# * **Specific demographic:** all patients are women aged 21+ of Pima Indian heritage, so the model may not generalise to men, other ethnic groups or other age ranges.
# * **Data quality:** many zeros in Insulin and SkinThickness are really missing values, and median imputation only approximates the true values.
# * **Limited features and a simple model:** no family history, diet, activity or lab trends, and logistic regression assumes a linear relationship in the log-odds.
# * **Educational benchmark, not a medical-grade tool:** it is useful for learning classification concepts but is not validated, calibrated or approved for clinical diagnosis.

# %% [markdown]
# ## 9. Final Summary Generator

# %%
summary = (
    f"This project built a Logistic Regression classifier on the Pima Indians Diabetes dataset "
    f"to predict diabetes (Outcome = 1). Invalid zeros in Glucose, BloodPressure, SkinThickness, "
    f"Insulin and BMI were treated as missing values and filled with the training-set median, "
    f"features were standardised, and a stratified 80/20 split (random_state=42) with "
    f"class_weight='balanced' was used to handle class imbalance. On the held-out test set the "
    f"model achieved an accuracy of {metrics['Accuracy']:.3f}, precision of {metrics['Precision']:.3f}, "
    f"recall of {metrics['Recall']:.3f}, an F1-score of {metrics['F1-score']:.3f} and a ROC-AUC of "
    f"{metrics['ROC-AUC']:.3f}, correctly detecting {tp} of {tp + fn} diabetic cases while missing {fn}. "
    f"{top2[0]} and {top2[1]} had the strongest positive coefficients, in line with their known clinical "
    f"role as risk factors. Recall is the key metric for screening because a missed case (false negative) "
    f"is costlier than a false alarm. However, the dataset is small and drawn from a specific population, "
    f"so the model is suitable for learning classification concepts and not for medical diagnosis."
)
import textwrap
print(textwrap.fill(summary, width=100))
