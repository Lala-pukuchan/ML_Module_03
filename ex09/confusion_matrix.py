import numpy as np
import pandas as pd


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
        instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    # error handling
    if not isinstance(y_true, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y_true.shape != y_hat.shape:
        return None

    # create labels from data
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_hat)))
    num_labels = len(labels)

    # Initialize confusion matrix
    confusion_matrix_ = np.zeros((num_labels, num_labels), dtype=int)

    # Create a label to index mapping
    label_to_index = {label: index for index, label in enumerate(labels)}

    y_true = y_true.flatten()
    y_hat = y_hat.flatten()

    # Populate the confusion matrix
    for true, pred in zip(y_true, y_hat):
        if true in label_to_index and pred in label_to_index:
            true_index = label_to_index[true]
            pred_index = label_to_index[pred]
            confusion_matrix_[true_index, pred_index] += 1

    # Convert to dataframe if df_option is True
    if df_option:
        return pd.DataFrame(confusion_matrix_, index=labels, columns=labels)

    return confusion_matrix_
