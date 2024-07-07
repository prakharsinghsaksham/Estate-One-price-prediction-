import numpy as np
import matplotlib.pyplot as plt

# Generate random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1) - 1  # Generate 100 random values between -1 and 1
y = 5 * X**2 + np.random.randn(100, 1)  # Quadratic relationship with noise

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.title('Generated Dataset')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
