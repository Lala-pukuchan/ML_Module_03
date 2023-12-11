import numpy as np


def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    # erro handling
    if not isinstance(y, np.ndarray) or y.shape[1] != 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.shape[1] != 1:
        return None

    m = y.shape[0]
    vec_one = np.ones(m).reshape(-1, 1)

    try:
        # calculate cost function with vectorization
        cost = -(1 / m) * (
            np.dot(y.T, np.log(y_hat + eps))
            + np.dot((vec_one - y).T, np.log(1 - y_hat + eps))
        )
        return cost[0, 0]
    except:
        return None
