import numpy as np
from ex01.log_pred import logistic_predict_


def log_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop.
    The three arrays must have compatibl Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    # error handling
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
    ):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None

    # initialize gradient vector
    gradient = np.zeros(theta.shape)

    # append 1 to x's first column
    x_dash = np.insert(x, 0, 1, axis=1)

    # compute hypothesis with sigmoid
    hypothesis = logistic_predict_(x, theta)

    # number of examples
    m = x.shape[0]

    # for loop for m examples
    for i in range(m):
        scalar_difference = hypothesis[i] - y[i]
        gradient += (scalar_difference) * x_dash[i].reshape(-1, 1)

    return gradient / m
