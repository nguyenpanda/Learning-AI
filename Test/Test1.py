import numpy as np
import matplotlib.pyplot as plt
import random
from MyLib import timeChecking


def squared_error(X, Y, a, b):
    lenX = len(X)
    lenY = len(Y)
    
    if lenX != lenY:
        raise ValueError("Input arrays X and Y must have the same length.")
    
    result = 0
    
    for i in range(lenX):
        result += (a * X[i] + b - Y[i]) ** 2
    
    return result / lenX

def gradient_descent(X, Y, a, b, learning_rate, num_iterations):
    num_samples = len(X)
    if num_samples != len(Y):
        raise ValueError("Input arrays X and Y must have the same length.")
    
    for i in range(num_iterations):
        # Calculate the gradient of the loss function with respect to a and b
        gradient_a = 0.0
        gradient_b = 0.0

        for j in range(num_samples):
            gradient_a += (a * X[j] + b - Y[j]) * X[j]
            gradient_b += (a * X[j] + b - Y[j])

        # Update a and b using the gradients
        a -= 2 * learning_rate * gradient_a / num_samples
        b -= 2 * learning_rate * gradient_b / num_samples

    return a, b


# Set a random seed for reproducibility
seed = 2
np.random.seed(seed)
random.seed(seed)

# Generate random data
num_samples = 100
X = np.random.rand(num_samples, 1) * 10  # Random input values (features)
Y = 2 * X + 1 + np.random.randn(num_samples, 1)  # Linear relationship with some noise

# Initialize a and b
a = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
num_iterations = 100

a, b = gradient_descent(X, Y, a, b, learning_rate, num_iterations)

# Print the optimized values of a and b
print(f"Optimized values: a = {a[0]}, b = {b[0]}")
print(f"MSE: {squared_error(X, Y, a, b)[0]}")

# Plot the data and the line
plt.scatter(X, Y, label='Data Points', marker='.')
plt.plot(X, a * X + b, label=f'{a[0]}*X + {b[0]}', color='blue')
plt.xlabel('X (Feature)')
plt.ylabel('Y (Target)')
plt.title('Sample Data for Linear Regression')
plt.legend()
plt.grid(True)
plt.axhline(0, color='red', linewidth=0.5)
plt.axvline(0, color='red', linewidth=0.5)
plt.show()



# # Gradient Descent
# for i in range(num_iterations):
#     # Calculate the gradient of the loss function with respect to a and b
#     gradient_a = 0.0
#     gradient_b = 0.0
#
#     for j in range(num_samples):
#         gradient_a += (a * X[j] + b - Y[j]) * X[j]
#         gradient_b += (a * X[j] + b - Y[j])
#
#     # Update a and b using the gradients
#     a -= learning_rate * gradient_a / num_samples
#     b -= learning_rate * gradient_b / num_samples

