import numpy as np
import matplotlib.pyplot as plt

# True values and predictions
y_true = np.array([3, 5, 2.5, 7])
y_pred_good = np.array([2.8, 5.2, 2.7, 6.9])
y_pred_bad = np.array([1, 8, 0, 10])

def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat)**2))

rmse_good = rmse(y_true, y_pred_good)
rmse_bad = rmse(y_true, y_pred_bad)

print(f"Good model RMSE: {rmse_good:.3f}")
print(f"Bad model RMSE: {rmse_bad:.3f}")

# Visualization
x = np.arange(len(y_true))
plt.scatter(x, y_true, label='True', color='blue', marker='o')
plt.scatter(x, y_pred_good, label=f'Good pred\nRMSE={rmse_good:.2f}', color='green', marker='x')
plt.scatter(x, y_pred_bad, label=f'Bad pred\nRMSE={rmse_bad:.2f}', color='red', marker='x')

for i in x:
    plt.plot([i, i], [y_true[i], y_pred_good[i]], color='green', alpha=0.3)
    plt.plot([i, i], [y_true[i], y_pred_bad[i]], color='red', alpha=0.3)

plt.xlabel("Sample")
plt.ylabel("Value")
plt.title("RMSE Visualization")
plt.legend()
plt.grid()
plt.show()