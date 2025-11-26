import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv(r"C:\Users\coccl\Desktop\ai-ml related stuff\dataset\cleaned\Car_cleaned_dataset.csv")

df.drop("Car_Name", axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

sample = X.iloc[[0]]          # Valid encoded sample
sample = scaler.transform(sample)
print("Predicted Car Price:", model.predict(sample)[0])

joblib.dump(model, "car_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")
