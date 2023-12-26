import numpy as np
from ex03.vec_log_loss import vec_log_loss_
from ex01.log_pred import logistic_predict_

y1 = np.array([1]).reshape((-1, 1))
x1 = np.array([4]).reshape((-1, 1))
theta1 = np.array([[2], [0.5]])
y_hat1 = logistic_predict_(x1, theta1)
print("\n-------------\n")
print("Example 1:")
print("vec_log_loss   :", vec_log_loss_(y1, y_hat1))
print("expected output:", 0.018149927917808714)

y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
y_hat2 = logistic_predict_(x2, theta2)
print("\n-------------\n")
print("\nExample 2:")
print("vec_log_loss   :", vec_log_loss_(y2, y_hat2))
print("expected output:", 2.4825011602472347)

y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
y_hat3 = logistic_predict_(x3, theta3)
print("\n-------------\n")
print("\nExample 3:")
print("vec_log_loss   :", vec_log_loss_(y3, y_hat3))
print("expected output:", 2.993853310859968)
