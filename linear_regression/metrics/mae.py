import numpy as np
import matplotlib.pyplot as plt

# Define the error values
e = np.linspace(-10, 10, 500)

# Loss functions
mse_loss = e**2
mae_loss = np.abs(e)

# Gradients
mse_grad = 2*e
mae_grad = np.sign(e)  # -1 for e<0, +1 for e>0, 0 for e=0

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss functions
axes[0].plot(e, mse_loss, label='MSE Loss', color='blue')
axes[0].plot(e, mae_loss, label='MAE Loss', color='red')
axes[0].set_title("Loss Functions")
axes[0].set_xlabel("Error (e)")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Gradients
axes[1].plot(e, mse_grad, label='MSE Gradient', color='blue')
axes[1].plot(e, mae_grad, label='MAE Gradient', color='red')
axes[1].set_title("Gradients of Loss Functions")
axes[1].set_xlabel("Error (e)")
axes[1].set_ylabel("Gradient")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()  