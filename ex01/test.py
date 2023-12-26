import numpy as np
from ex01.log_pred import logistic_predict_

# Example 1
x = np.array([4]).reshape((-1, 1))
theta = np.array([[2], [0.5]])
print("\n-------------\n")
print("my logistic_predict:\n", logistic_predict_(x, theta))
# Output:
print("expected:\n", np.array([[0.98201379]]))

print("\n-------------\n")
# Example 1
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print("my logistic_predict (Example 1):\n", logistic_predict_(x2, theta2))
# Expected Output
print(
    "expected (Example 1):\n",
    np.array([[0.98201379], [0.99624161], [0.97340301], [0.99875204], [0.90720705]]),
)

print("\n-------------\n")
# Example 2
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print("my logistic_predict (Example 2):\n", logistic_predict_(x3, theta3))
# Expected Output
print("expected (Example 2):\n", np.array([[0.03916572], [0.00045262], [0.2890505]]))
