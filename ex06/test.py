import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

# Test data
X = np.array([[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [3.0, 5.0, 9.0, 14.0]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])

# Initialize MyLogisticRegression with thetas
mylr = MyLR(thetas)

# Example 0: Predict
print("Example 0:")
predicted_0 = mylr.predict_(X)
print("Predicted:\n", predicted_0)
print("Expected:\n", np.array([[0.99930437], [1.0], [1.0]]))

# Example 1: Loss
print("\nExample 1:")
loss_1 = mylr.loss_(X, Y)  # should be y and y_hat???
print("Loss:", loss_1)
print("Expected Loss:", 11.513157421577004)

# Example 2: Fit and updated theta
print("\nExample 2:")
mylr.fit_(X, Y)
updated_theta_2 = mylr.theta
print("Updated Theta:\n", updated_theta_2)
print(
    "Expected Theta:\n",
    np.array([[2.11826435], [0.10154334], [6.43942899], [-5.10817488], [0.6212541]]),
)

# Example 3: Predict after fitting
print("\nExample 3:")
predicted_3 = mylr.predict_(X)
print("Predicted after fitting:\n", predicted_3)
print("Expected:\n", np.array([[0.57606717], [0.68599807], [0.06562156]]))

# Example 4: Loss after fitting
print("\nExample 4:")
loss_4 = mylr.loss_(X, Y)
print("Loss after fitting:", loss_4)
print("Expected Loss after fitting:", 1.4779126923052268)
