import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from ex06.my_logistic_regression import MyLogisticRegression as MyLR
from ex07.mono_log import data_spliter


def multi_log():
    # Load dataset for y
    solar_system_census_planets = pd.read_csv("ex07/solar_system_census_planets.csv")

    # Load dataset for x
    solar_system_census = pd.read_csv("ex07/solar_system_census.csv")
    x = np.array(solar_system_census[["weight", "height", "bone_density"]])

    # save predicted probabilities
    predicted_probabilities_dict = {}
    for zipcode in range(0, 5):
        # Generate new labels with if it's selected zip code or not
        y = np.where(solar_system_census_planets["Origin"] == zipcode, 1, 0)
        y = y.reshape(-1, 1)

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

        # Evaluate accuracy of the model
        predicted_probabilities_dict[zipcode] = model.predict_(x_test)

    # Stack the probabilities horizontally
    all_probabilities = np.column_stack(
        [predicted_probabilities_dict[zipcode] for zipcode in range(5)]
    )

    # Print the first few rows of all_probabilities for checking
    print("First few rows of all_probabilities:\n", all_probabilities[:5])

    # Find the class (zipcode) with the highest probability for each example
    predicted_classes = np.argmax(all_probabilities, axis=1)

    # Print the first few predicted classes for checking
    print("First few predicted classes:\n", predicted_classes[:5])

    # Calculate and display the accuracy
    y = np.array(solar_system_census_planets["Origin"])
    y = y.reshape(-1, 1)

    # Split data
    predicted_classes = predicted_classes.astype(int)
    correct_predictions = np.sum(y.flatten() == predicted_classes)
    print("y",y.flatten())
    print("predicted_classes",predicted_classes)
    total_predictions = len(y)
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)

    # Plotting
    feature_pairs = [
        ("weight", "height"),
        ("weight", "bone_density"),
        ("height", "bone_density"),
    ]
    feature_indices = {"weight": 0, "height": 1, "bone_density": 2}

    for pair in feature_pairs:
        feature_index_1 = feature_indices[pair[0]]
        feature_index_2 = feature_indices[pair[1]]

        plt.figure(figsize=(8, 6))
        for class_index in range(5):
            mask = predicted_classes == class_index
            plt.scatter(x_test[mask, feature_index_1],
                        x_test[mask, feature_index_2],
                        label=f"Predicted Class {class_index}", alpha=0.5)
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.title(f"{pair[0]} vs {pair[1]}: Predicted Classes")
        plt.legend()
        plt.savefig(f"results/ex07/multi_figure_{pair[0]}_vs_{pair[1]}.png")
        plt.close()



def main():
    multi_log()


if __name__ == "__main__":
    main()
