import numpy as np


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    # error handling
    if not isinstance(y, np.ndarray) or y.shape[1] != 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.shape[1] != 1:
        return None
    if not isinstance(eps, float):
        return None

    try:
        # calculate cost function
        m = y.shape[0]
        cost = -(1 / m) * (
            np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        )
        return cost
    except:
        return None
