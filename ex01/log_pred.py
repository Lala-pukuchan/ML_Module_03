import numpy as np
from ex00.sigmoid import sigmoid_


def logistic_predict_(x, theta):
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    # error handling
    if not isinstance(x, np.ndarray):
        return None
    if not isinstance(theta, np.ndarray) or theta.shape[1] != 1:
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if x.shape[1] + 1 != theta.shape[0]:
        return None

    # append 1 to x's first column
    x_dash = np.insert(x, 0, 1, axis=1)

    # compute y_hat
    y_hat = sigmoid_(np.dot(x_dash, theta))
    return y_hat
