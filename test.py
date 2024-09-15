import numpy as np

def f(x):
    return abs(10 - (0.088*x[0] + 0.24*x[1] + 0.2*x[2]))

def gradient(x):
    if 10 - (0.088*x[0] + 0.24*x[1] + 0.2*x[2]) >= 0:
        return np.array([-0.088, -0.24, -0.2])
    else:
        return np.array([0.088, 0.24, 0.2])

def gradient_descent(learning_rate=0.001, num_iterations=1000):
    x = np.zeros(3)  # Initial guess [A, B, C]
    for _ in range(num_iterations):
        grad = gradient(x)
        x -= learning_rate * grad
    return x

result = gradient_descent()
print(f"A = {result[0]}, B = {result[1]}, C = {result[2]}")
print(f"Minimum value: {f(result)}")
