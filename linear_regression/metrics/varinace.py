import numpy as np
import matplotlib.pyplot as plt

# 1. Generate data
np.random.seed(0)
x_low_var = np.random.normal(10, 1, 100)   # mean=10, std=1
x_high_var = np.random.normal(10, 5, 100)  # mean=10, std=5

# 2. Compute variance
def compute_variance(x):
    mean = np.mean(x)
    var = np.mean((x - mean)**2)
    return var

var_low = compute_variance(x_low_var)
var_high = compute_variance(x_high_var)

# numpy for verification
var_low_np = np.var(x_low_var)
var_high_np = np.var(x_high_var)

print(f"Low variance dataset: manual={var_low:.2f}, numpy={var_low_np:.2f}")
print(f"High variance dataset: manual={var_high:.2f}, numpy={var_high_np:.2f}")

# 3. Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].hist(x_low_var, bins=15, color='blue', alpha=0.7)
axes[0].axvline(np.mean(x_low_var), color='red', linestyle='--', label='Mean')
axes[0].set_title(f"Low Variance: {var_low:.2f}")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].hist(x_high_var, bins=15, color='green', alpha=0.7)
axes[1].axvline(np.mean(x_high_var), color='red', linestyle='--', label='Mean')
axes[1].set_title(f"High Variance: {var_high:.2f}") 
axes[1].set_xlabel("Value")
axes[1].legend()

plt.tight_layout()
plt.show()