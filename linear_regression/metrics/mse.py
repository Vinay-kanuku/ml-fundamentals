import numpy as np
import matplotlib.pyplot as plt

# Simulated data
x = np.linspace(0, 10, 50)
y = 3*x + np.random.randn(50) * 2  # true slope 3 + noise

# Define MSE and gradient
def mse(w):
    y_pred = w * x
    return np.mean((y - y_pred)**2)

def grad_mse(w):
    y_pred = w * x
    return -2 * np.mean(x * (y - y_pred))

# Visualize MSE curve
w_vals = np.linspace(0, 6, 100)
mse_vals = [mse(w) for w in w_vals]
grad_vals = [grad_mse(w) for w in w_vals]

plt.figure(figsize=(12,5))

# Loss curve
plt.subplot(1,2,1)
plt.plot(w_vals, mse_vals)
plt.title("MSE vs weight")
plt.xlabel("w") 
plt.ylabel("MSE")

# Gradient
plt.subplot(1,2,2)
plt.plot(w_vals, grad_vals)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Gradient of MSE vs weight")
plt.xlabel("w")

plt.ylabel("Gradient")

plt.show()