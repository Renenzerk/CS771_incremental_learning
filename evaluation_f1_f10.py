# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tensorflow as tf
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import os
import time
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class PrototypeClassifier:
    """
    A simple prototype-based classifier that uses the nearest prototype
    for classification. Each class is represented by a single prototype
    that is updated incrementally during training.
    """

    def __init__(self):
        """
        Initialize the classifier with empty prototypes and class counts.
        """
        self.prototypes = {}  # Maps class labels to their prototype feature vectors
        self.class_counts = {}  # Maps class labels to the count of samples assigned to each class

    def euclidean_distance(self, x, y):
        """
        Calculate the Euclidean distance between two vectors.
        Args:
            x (numpy.ndarray): First vector.
            y (numpy.ndarray): Second vector.
        Returns:
            float: Euclidean distance.
        """
        return np.linalg.norm(x - y)  
        
    def train(self, features, labels):
        """
        Train the classifier by computing and updating class prototypes.
        Args:
            features (numpy.ndarray): Feature vectors for training data.
            labels (numpy.ndarray): Corresponding labels for training data.
        """
        for feature, label in zip(features, labels):  # Iterate over features and labels
            if label not in self.prototypes:
                # Initialize prototype and count for new class
                self.prototypes[label] = feature
                self.class_counts[label] = 1
            else:
                # Update existing prototype by averaging with the new feature
                self.prototypes[label] = (
                    self.prototypes[label] * self.class_counts[label] + feature
                ) / (self.class_counts[label] + 1)
                self.class_counts[label] += 1  # Increment the sample count for the class

    def predict(self, features):
        """
        Predict class labels for a set of feature vectors.
        Args:
            features (numpy.ndarray): Feature vectors.
        Returns:
            numpy.ndarray: Predicted labels.
        """
        predictions = []
        for feature in features:  # Iterate over features
            # Compute distances from the feature to all prototypes
            distances = {
                label: self.euclidean_distance(feature, prototype)
                for label, prototype in self.prototypes.items()
            }
            # Find the label of the nearest prototype
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)  # Return predictions as a numpy array

    def update_classifier(self, features, pseudo_labels):
        """
        Incrementally update the classifier with new data and pseudo-labels.
        Args:
            features (numpy.ndarray): Feature vectors for the new data.
            pseudo_labels (numpy.ndarray): Pseudo-labels for the data
        """
        for feature, pseudo_label in zip(features, pseudo_labels):  # Iterate over features and pseudo-labels
            if pseudo_label not in self.prototypes:
                # Initialize prototype and count for a new pseudo-class
                self.prototypes[pseudo_label] = feature
                self.class_counts[pseudo_label] = 1
            else:
                # Update prototype for an existing pseudo-class
                self.prototypes[pseudo_label] = (
                    self.prototypes[pseudo_label] * self.class_counts[pseudo_label] + feature
                ) / (self.class_counts[pseudo_label] + 1)
                self.class_counts[pseudo_label] += 1  # Increment the count for the pseudo-class


def load_model(path):
    """
    Load a Python object (e.g., a trained model) from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        object: The loaded Python object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def task1_test_features():
    """
    Evaluate saved classifiers (f1 to f10) on corresponding test datasets (D1 to D10).
    - Each model is tested on its corresponding datasets up to the training iteration (e.g., f5 is tested on D1 to D5).
    - Accuracy results are recorded in a matrix.
    - Log time taken for each testing step.
    - Plot the accuracy matrix as a heatmap.

    """
    # Directory containing the pre-extracted feature datasets for evaluation
    feature_dir = "part_one_dataset/eb3_extracted_features/eval"
    timings = []  # List to store time taken for each model's evaluation
    accuracy_matrix = np.zeros((10, 10))  # Initialize a 10x10 accuracy matrix

    # Loop through models f1 to f10
    for i in range(1, 11):
        start_time = time.time()  # Start timer for the testing step

        # Load the saved classifier for model f{i}
        model_path = f"models/f{i}.pkl"
        classifier = load_model(model_path)

        # Loop through datasets D1 to D{i} for evaluation
        for j in range(1, i + 1):
            # Path to the pre-extracted features for dataset D{j}
            feature_path = os.path.join(feature_dir, f"{j}_eval_features.npy")

            # Load the test features and labels for the dataset
            features = np.load(feature_path)
            labels = torch.load(f'dataset/dataset/part_one_dataset/eval_data/{j}_eval_data.tar.pth')['targets']

            # Predict labels using the trained classifier
            predictions = classifier.predict(features)

            # Calculate accuracy for the dataset and store it in the matrix
            accuracy = accuracy_score(labels, predictions) * 100
            accuracy_matrix[i - 1, j - 1] = accuracy
            # print(f"Model f{i}, Dataset D{j} - Accuracy: {accuracy:.2f}%")

        # End timer and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        timings.append((i, elapsed_time))
        print(f"Time taken to test model f{i}: {elapsed_time:.2f} seconds")
        
    return accuracy_matrix, timings


def heat_map(accuracy_matrix):
# Plotting the accuracy matrix as a heatmap
    plt.figure(figsize=(10, 8))  # Set figure size for better readability
    sns.heatmap(
        accuracy_matrix,
        annot=True,  # Annotate each cell with its accuracy value
        fmt=".2f",  # Format to two decimal places
        # cmap="viridis",  # Use a visually appealing color map
        xticklabels=[f"D{j}" for j in range(1, 11)],  # Label x-axis with datasets
        yticklabels=[f"f{i}" for i in range(1, 11)],  # Label y-axis with models
    )
    plt.title("Accuracy Matrix for Test Data")  # Add a title to the plot
    plt.xlabel("Dataset")  # Label the x-axis
    plt.ylabel("Model")  # Label the y-axis
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    
accuracy_matrix,timings=task1_test_features()
heat_map(accuracy_matrix)