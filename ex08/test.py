import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_


# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

# Accuracy
print("Accuracy - My implementation:", accuracy_score_(y, y_hat))
print("Accuracy - sklearn Implementation:", accuracy_score(y, y_hat))

# Precision
print("Precision - My implementation:", precision_score_(y, y_hat))
print("Precision - sklearn Implementation:", precision_score(y, y_hat))

# Recall
print("Recall - My implementation:", recall_score_(y, y_hat))
print("Recall - sklearn Implementation:", recall_score(y, y_hat))

# F1-score
print("F1 Score - My implementation:", f1_score_(y, y_hat))
print("F1 Score - sklearn Implementation:", f1_score(y, y_hat))

# Example 2:
y_hat = np.array(
    ["norminet", "dog", "norminet", "norminet", "dog", "dog", "dog", "dog"]
)
y = np.array(
    ["dog", "dog", "norminet", "norminet", "dog", "norminet", "dog", "norminet"]
)

# Accuracy
print("Accuracy - My implementation:", accuracy_score_(y, y_hat))
print("Accuracy - sklearn Implementation:", accuracy_score(y, y_hat))

# Precision
print("Precision - My implementation:", precision_score_(y, y_hat, pos_label="dog"))
print("Precision - sklearn Implementation:", precision_score(y, y_hat, pos_label="dog"))

# Recall
print("Recall - My implementation:", recall_score_(y, y_hat, pos_label="dog"))
print("Recall - sklearn Implementation:", recall_score(y, y_hat, pos_label="dog"))

# F1-score
print("F1 Score - My implementation:", f1_score_(y, y_hat, pos_label="dog"))
print("F1 Score - sklearn Implementation:", f1_score(y, y_hat, pos_label="dog"))

# Example 3:
y_hat = np.array(
    ["norminet", "dog", "norminet", "norminet", "dog", "dog", "dog", "dog"]
)
y = np.array(
    ["dog", "dog", "norminet", "norminet", "dog", "norminet", "dog", "norminet"]
)

# Precision for 'norminet'
print(
    "Precision ('norminet') - My implementation:",
    precision_score_(y, y_hat, pos_label="norminet"),
)
print(
    "Precision ('norminet') - sklearn Implementation:",
    precision_score(y, y_hat, pos_label="norminet"),
)

# Recall for 'norminet'
print(
    "Recall ('norminet') - My implementation:",
    recall_score_(y, y_hat, pos_label="norminet"),
)
print(
    "Recall ('norminet') - sklearn Implementation:",
    recall_score(y, y_hat, pos_label="norminet"),
)

# F1-score for 'norminet'
print(
    "F1 Score ('norminet') - My implementation:",
    f1_score_(y, y_hat, pos_label="norminet"),
)
print(
    "F1 Score ('norminet') - sklearn Implementation:",
    f1_score(y, y_hat, pos_label="norminet"),
)
