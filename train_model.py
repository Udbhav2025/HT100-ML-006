import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data directly from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

print("Downloading and loading dataset...")
df = pd.read_csv(url, names=column_names)

# 2. Preprocessing
# The dataset uses '?' for missing values. Replace with NaN.
df.replace('?', np.nan, inplace=True)

# Convert columns to numeric (errors='coerce' turns non-numbers into NaN)
for col in ['ca', 'thal']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Separate Features (X) and Target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Convert target to Binary: 0 = No Disease, 1+ = Disease
y = (y > 0).astype(int)

# 3. Handle Missing Data (Imputation)
# This fulfills the "Work accurately even with missing patient data" requirement
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 4. Train Model
print("Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Trained! Accuracy: {acc:.2%}")

# 5. Save the Model and the Imputer
# We need the imputer to handle missing values in the live app
joblib.dump(model, 'heart_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
print("Model and Imputer saved to disk.")