"""Download the Pima Indians Diabetes dataset and save it as diabetes.csv."""
import pandas as pd

COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

if __name__ == "__main__":
    df = pd.read_csv(URL, header=None, names=COLS)
    df.to_csv("diabetes.csv", index=False)
    print(f"Saved diabetes.csv with shape {df.shape}")
