import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if x.size == 0:
        return None

    return 1 / (1 + np.exp(-x))


np.set_printoptions(precision=17)


# Example 1:
x1 = np.array([[-4]]).reshape((-1, 1))
print("- Sigmoid Example 1:")
print("Calculated:", sigmoid_(x1))
print("Expected: [[0.01798620996209156]]")

# Example 2:
x2 = np.array([[2]]).reshape((-1, 1))
print("- Sigmoid Example 2:")
print("Calculated:", sigmoid_(x2))
print("Expected: [[0.8807970779778823]]")

# Example 3:
x3 = np.array([[-4], [2], [0]]).reshape((-1, 1))
print("- Sigmoid Example 3:")
print("Calculated:", sigmoid_(x3))
print("Expected: [[0.01798620996209156], [0.8807970779778823], [0.5]]")
