import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# LOAD YOUR PREPARED DATA
# ==============================
X = pd.read_csv("data/prepared/X.csv").values
y = pd.read_csv("data/prepared/y.csv").values

# ==============================
# FIXED SPLIT (IMPORTANT)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODEL 1: MultiOutput RF
# ==============================
rf_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=500, random_state=42)
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ==============================
# MODEL 2: SIMPLE STACKED MODEL
# ==============================
rf_base = RandomForestRegressor(n_estimators=200, random_state=42)
rf_base.fit(X_train, y_train)

train_pred = rf_base.predict(X_train)
meta_model = LinearRegression()
meta_model.fit(train_pred, y_train)

test_pred = rf_base.predict(X_test)
stack_pred = meta_model.predict(test_pred)

# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

rf_metrics = evaluate(y_test, rf_pred)
stack_metrics = evaluate(y_test, stack_pred)

# ==============================
# PRINT RESULTS
# ==============================
print("\n===== FINAL COMPARISON =====")
print("\nMultiOutput Random Forest:")
print(f"MAE: {rf_metrics[0]:.2f}")
print(f"RMSE: {rf_metrics[1]:.2f}")
print(f"R2: {rf_metrics[2]:.4f}")

print("\nStacked Model:")
print(f"MAE: {stack_metrics[0]:.2f}")
print(f"RMSE: {stack_metrics[1]:.2f}")
print(f"R2: {stack_metrics[2]:.4f}")
import matplotlib.pyplot as plt
import numpy as np

# Your REAL values
rf_vals = [5.74, 7.12, 0.8427]
stack_vals = [6.45, 8.12, 0.7935]

metrics = ['MAE', 'RMSE', 'R²']

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(7,5))

plt.bar(x - width/2, rf_vals, width, label='MultiOutput RF')
plt.bar(x + width/2, stack_vals, width, label='Stacked Model')

plt.xticks(x, metrics)
plt.ylabel("Values")
plt.title("Model Comparison (Same Dataset)")

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=200)
plt.show()