import numpy as np
from confusion_matrix import confusion_matrix_
from sklearn.metrics import confusion_matrix

# Test data
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird']).reshape(-1, 1)
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet']).reshape(-1, 1)

# Example 1:
print("Example 1:")
print("Your implementation:\n", confusion_matrix_(y, y_hat))
print("sklearn implementation:\n", confusion_matrix(y, y_hat))

# Example 2:
print("\nExample 2:")
print("Your implementation:\n", confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
print("sklearn implementation:\n", confusion_matrix(y, y_hat, labels=['dog', 'norminet']))

# Example 3:
print("\nExample 3:")
print("Your implementation (DataFrame):\n", confusion_matrix_(y, y_hat, df_option=True))

# Example 4:
print("\nExample 4:")
print("Your implementation (DataFrame with specified labels):\n", confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
