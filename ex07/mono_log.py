import numpy as np
import sys
import pandas as pd
from ex06.my_logistic_regression import MyLogisticRegression as MyLR
import matplotlib.pyplot as plt


def data_spliter(x, y, proportion):
    """
    Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
      x: numpy.array, a matrix of dimension m * n.
      y: numpy.array, a vector of dimension m * 1.
      proportion: float, the proportion of the dataset that will be assigned to the training set.
    Return:
      (x_train, x_test, y_train, y_test) as a tuple of numpy.array
      None if x or y is an empty numpy.array.
      None if x and y do not share compatible dimensions.
      None if x, y or proportion is not of expected type.
    Raises:
      This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(proportion, float)
    ):
        return None
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return None

    # Combine x and y and shuffle
    combined = np.hstack((x, y.reshape(-1, 1)))
    np.random.shuffle(combined)

    # Split the combined array back into x and y
    split_idx = int(combined.shape[0] * proportion)
    x_train = combined[:split_idx, :-1]
    x_test = combined[split_idx:, :-1]
    y_train = combined[:split_idx, -1:]
    y_test = combined[split_idx:, -1:]

    return (x_train, x_test, y_train, y_test)


def mono_log(zipcode):
    # Your function logic here
    print(f"Function executed with zipcode: {zipcode}")

    # convert to follow dataset format
    zipcode = float(zipcode)

    # Load dataset for y
    solar_system_census_planets = pd.read_csv("ex07/solar_system_census_planets.csv")

    # Generate new labels with if it's selected zip code or not
    y = np.where(solar_system_census_planets["Origin"] == zipcode, 1, 0)
    y = y.reshape(-1, 1)

    # Load dataset for x
    solar_system_census = pd.read_csv("ex07/solar_system_census.csv")
    x = np.array(solar_system_census[["weight", "height", "bone_density"]])

    # Split data
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    # Scale data
    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    x_train = (x_train - x_min) / (x_max - x_min)
    x_test = (x_test - x_min) / (x_max - x_min)

    # Initialize the model
    theta = np.zeros((x_train.shape[1] + 1, 1))
    alpha = 1e-2
    max_iter = 10000
    model = MyLR(theta, alpha, max_iter)

    # Fit the model
    model.fit_(x_train, y_train)
    print("theta:\n", model.theta)

    # Evaluate mse of the model
    train_mse = model.loss_(x_train, y_train)
    print("train_mse:\n", train_mse)
    test_mse = model.loss_(x_test, y_test)
    print("test_mse:\n", test_mse)

    # Evaluate accuracy of the model
    predicted_probabilities = model.predict_(x_test)
    # print("predicted_probabilities:", predicted_probabilities)
    predicted_labels = (predicted_probabilities >= 0.5).astype(int)
    # print("predicted_labels:", predicted_labels)
    correct_predictions = np.sum(y_test == predicted_labels)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    print("Correct Predictions:\n", correct_predictions)
    print("Total Predictions:\n", total_predictions)
    print("Accuracy:\n", accuracy)

    # plotting
    feature_pairs = [
        ("weight", "height"),
        ("weight", "bone_density"),
        ("height", "bone_density"),
    ]
    feature_indices = {"weight": 0, "height": 1, "bone_density": 2}

    for pair in feature_pairs:
        feature_index_1 = feature_indices[pair[0]]
        feature_index_2 = feature_indices[pair[1]]

        plt.figure(figsize=(12, 6))

        # Plot for Actual Zip Code
        plt.subplot(1, 2, 1)
        plt.scatter(
            x_test[:, feature_index_1],
            x_test[:, feature_index_2],
            c=y_test.flatten(),
            cmap="winter",
            edgecolor="k",
            s=50,
        )
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.title(f"Actual Zip Code: {pair[0]} vs {pair[1]}")

        # Plot for Predicted Zip Code
        plt.subplot(1, 2, 2)
        plt.scatter(
            x_test[:, feature_index_1],
            x_test[:, feature_index_2],
            c=predicted_labels.flatten(),
            cmap="autumn",
            edgecolor="k",
            s=50,
        )
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.title(f"Predicted Zip Code: {pair[0]} vs {pair[1]}")

        plt.suptitle(f"Actual vs Predicted Zip Code Comparison: {pair[0]} vs {pair[1]}")
        plt.savefig(f"results/ex07/mono_figure_{pair[0]}_vs_{pair[1]}.png")
        plt.close()


def main():
    # Check if the argument is provided
    if len(sys.argv) < 2:
        print("Usage: test.py --zipcode=<0|1|2|3>")
        sys.exit(1)  # Exit the script if no argument is provided

    # Retrieve the argument
    arg = sys.argv[1]

    # Check if the argument is in the correct format
    if arg.startswith("--zipcode="):
        zipcode = arg.split("=")[1]
        if zipcode in ["0", "1", "2", "3"]:
            # Call your function with the zipcode
            mono_log(zipcode)
        else:
            print("Invalid zipcode. Usage: test.py --zipcode=<0|1|2|3>")
            sys.exit(1)
    else:
        print("Invalid argument. Usage: test.py --zipcode=<0|1|2|3>")
        sys.exit(1)


if __name__ == "__main__":
    main()
