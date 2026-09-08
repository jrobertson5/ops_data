from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# Locate files relative to this Python script
project_directory = Path(__file__).resolve().parent

csv_path = project_directory / "wdc_productivity.csv"
model_path = project_directory / "pack_orders_model.joblib"


# Read the source data
df_prod = pd.read_csv(csv_path)


# Columns used to make predictions
features = [
    "pick_units",
    "pick_lh",
    "put_units",
    "put_lh",
    "order_lh"
]

target = "pack_orders"


# Keep the required columns and remove incomplete rows
model_data = df_prod[features + [target]].dropna()

X = model_data[features]
y = model_data[target]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Evaluate the model on data it did not train on
test_predictions = model.predict(X_test)

test_r2 = r2_score(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

print("Model coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(f"  {feature}: {coefficient:.4f}")

print(f"Intercept: {model.intercept_:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Mean absolute error: {test_mae:.2f}")


# Save both the model and its required feature list
model_bundle = {
    "model": model,
    "features": features
}

joblib.dump(model_bundle, model_path)

print(f"\nModel saved to:\n{model_path}")