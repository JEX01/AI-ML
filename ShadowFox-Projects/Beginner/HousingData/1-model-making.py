import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\coccl\Desktop\ai-ml related stuff\dataset\cleaned\HousingData_cleaned_dataset.csv")

# Features & Target
X = df.drop("MEDV", axis=1)   # All input features
y = df["MEDV"]               # House price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

sample = pd.DataFrame([[0.1, 12, 5.2, 0, 0.45, 6.5, 45, 4.2, 4, 310, 17, 390, 8]], columns=X.columns)
sample = scaler.transform(sample)
print("Predicted House Price:", model.predict(sample)[0])

import joblib

joblib.dump(model, "boston_house_model.pkl")
joblib.dump(scaler, "scaler.pkl")   # save scaler too (important)

print("Model and scaler saved successfully.")
