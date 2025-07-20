import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 3*X.flatten() + 5 + np.random.randn(50) * 5  # linear + noise

# 2. Fit Linear Regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 3. Compute R² manually
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean)**2)
ss_res = np.sum((y - y_pred)**2)
r2_manual = 1 - (ss_res / ss_tot)

# Also compute R² with sklearn for verification
r2_sklearn = r2_score(y, y_pred)

print(f"Manual R²:   {r2_manual:.4f}")
print(f"Sklearn R²:  {r2_sklearn:.4f}")

# 4. Visualize
plt.figure(figsize=(8,6))
plt.scatter(X, y, label='Data', color='blue')
plt.plot(X, y_pred, label=f'Model Prediction\n$R^2$ = {r2_manual:.2f}', color='red')
plt.axhline(y_mean, color='gray', linestyle='--', label='Mean of y') # type: ignore

# Residuals (optional: show errors)
for xi, yi, ypi in zip(X.flatten(), y, y_pred):
    plt.plot([xi, xi], [yi, ypi], color='gray', alpha=0.3)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit and $R^2$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()