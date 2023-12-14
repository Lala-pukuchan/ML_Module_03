import numpy as np
from ex01.log_pred import logistic_predict_


def vec_log_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop.
    The three arrays must have comp Args:
       x: has to be an numpy.ndarray, a matrix of shape m * n.
       y: has to be an numpy.ndarray, a vector of shape m * 1.
       theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
       The gradient as a numpy.ndarray, a vector of shape n * 1,
       containg the result of the formula for all j.
       None if x, y, or theta are empty numpy.ndarray.
       None if x, y and theta do not have compatible shapes.
    Raises:
       This function should not raise any Exception.
    """
    # input validation
    # if x, y and theta are not numpy.ndarray, return None
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
    ):
        return None

    # if x, y and theta are empty numpy.ndarray, return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None

    # if x, y and theta do not have compatible shapes, return None
    if not (x.shape[0] == y.shape[0] and x.shape[1] + 1 == theta.shape[0]):
        return None

    # number of examples
    m = x.shape[0]

    # insert 1 to the first column of x
    x_dash = np.insert(x, 0, 1, axis=1)

    # compute hypothesis with sigmoid
    hypothesis = logistic_predict_(x, theta)

    # return gradient vector
    return (1 / m) * np.dot(x_dash.T, hypothesis - y)
