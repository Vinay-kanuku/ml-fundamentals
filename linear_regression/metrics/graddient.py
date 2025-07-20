
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 + 2 * X + np.random.randn(100, 1)

# Add bias feature
X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

# Initialize weights

w = np.zeros((2, 1))
lr = 0.1


n_iters = 20
m = len(y)
# Record history
w_history = []
loss_history = []

for i in range(n_iters):
    preds = X_bias @ w
    error = preds - y
    gradients = (2/m) * X_bias.T @ error
    w -= lr * gradients

    w_history.append(w.flatten().copy())
    loss = np.mean(error**2)
    loss_history.append(loss)
 
w_history = np.array(w_history)

print(f"Final weights: {w.ravel()}")

# Prepare loss surface
w0_range = np.linspace(-1, 6, 100)
w1_range = np.linspace(0, 4, 100)
W0, W1 = np.meshgrid(w0_range, w1_range)

def compute_loss(w0, w1):
    weights = np.array([[w0], [w1]])
    preds = X_bias @ weights
    return np.mean((y - preds)**2)

Z = np.array([[compute_loss(w0, w1) for w0 in w0_range] for w1 in w1_range])

# Plot
plt.figure(figsize=(10, 6))
cp = plt.contour(W0, W1, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.xlabel(r"$w_0$ (bias)")
plt.ylabel(r"$w_1$ (slope)")
plt.title("Gradient Descent Path on MSE Loss Surface")

# GD path
plt.plot(w_history[:, 0], w_history[:, 1], "r.-", label="GD path")
plt.legend()
plt.show()

# Optional: Loss vs iteration
plt.figure()
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Convergence of Loss")
plt.grid(True)
plt.show()