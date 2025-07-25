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
        # predictions = []
        # for feature in features:  # Iterate over features
        #     # Compute distances from the feature to all prototypes
        #     distances = {
        #         label: self.euclidean_distance(feature, prototype)
        #         for label, prototype in self.prototypes.items()
        #     }
        #     # Find the label of the nearest prototype
        #     predictions.append(min(distances, key=distances.get))
        # return np.array(predictions)  # Return predictions as a numpy array
        predictions = []
        confidences = []
        beta=1.0
        for feature in features:
            distances = {
                label: self.euclidean_distance(feature, prototype)
                for label, prototype in self.prototypes.items()
            }
            # Softmax over negative distances
            labels = list(distances.keys())
            dists = np.array(list(distances.values()))
            scores = np.exp(-beta * dists)
            probs = scores / np.sum(scores)
            
            # Get predicted label and its confidence
            best_idx = np.argmax(probs)
            predicted_label = labels[best_idx]
            confidence = probs[best_idx]
            
            predictions.append(predicted_label)
            confidences.append(confidence)
            
        return np.array(predictions), np.array(confidences)

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
                alpha=0.4
                # Update prototype for an existing pseudo-class
                self.prototypes[pseudo_label] = (
                    self.prototypes[pseudo_label] * self.class_counts[pseudo_label] + feature
                ) / (self.class_counts[pseudo_label] + 1)
                self.class_counts[pseudo_label] += 1  # Increment the count for the pseudo-class
                # self.prototypes[pseudo_label] = (
                # (1 - alpha) * self.prototypes[pseudo_label] + alpha * feature)


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
    Evaluate the saved classifiers on pre-extracted test datasets (D1 to D20).
    Log the time taken for each testing step and plot the accuracy matrix.
    """
    # Paths to pre-extracted evaluation features for Part 1 and Part 2 datasets
    feature_dir1 = "part_one_dataset/eb3_extracted_features/eval"
    feature_dir2 = "part_two_dataset/eb3_extracted_features/eval"

    timings = []  # List to store the time taken for testing each model
    accuracy_matrix = np.zeros((10, 20))  # Initialize a 10x20 accuracy matrix

    # Loop through models f11 to f20
    for i in range(11, 21):
        start_time = time.time()  # Start timer for testing the current model

        # Load the trained classifier model f{i}
        model_path = f"models/f{i}.pkl"
        classifier = load_model(model_path)

        # Evaluate the model on all datasets D1 to D{i}
        for j in range(1, i+1):
            if j <= 10:
                # Load Part 1 evaluation features and labels
                feature_path = os.path.join(feature_dir1, f"{j}_eval_features.npy")
                labels = torch.load(f'dataset/dataset/part_one_dataset/eval_data/{j}_eval_data.tar.pth')['targets']
            else:
                # Load Part 2 evaluation features and labels
                feature_path = os.path.join(feature_dir2, f"{j - 10}_eval_features.npy")
                labels = torch.load(f'dataset/dataset/part_two_dataset/eval_data/{j - 10}_eval_data.tar.pth')['targets']

            # Load pre-extracted features
            features = np.load(feature_path)

            # Predict class labels using the current classifier
            predictions,confidences = classifier.predict(features)

            # Calculate accuracy for the current dataset and store in the matrix
            accuracy = accuracy_score(labels, predictions) * 100
            accuracy_matrix[i - 11, j - 1] = accuracy
            # print(f"Model f{i}, Dataset D{j} - Accuracy: {accuracy:.2f}%")

        # End timer and log the elapsed time for testing the current model
        end_time = time.time()
        elapsed_time = end_time - start_time
        timings.append((i, elapsed_time))
        print(f"Time taken to test model f{i}: {elapsed_time:.2f} seconds")
    return accuracy_matrix
    

def heat_map(accuracy_matrix):
    # Plot the accuracy matrix as a heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt=".2f",
        # cmap="viridis",
        xticklabels=[f"D{j}" for j in range(1, 21)],  # Dataset labels
        yticklabels=[f"f{i}" for i in range(11, 21)],  # Model labels
    )
    plt.title("Accuracy Matrix for Test Data")
    plt.xlabel("Dataset")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()
    
accuracy_matrix=task1_test_features()
heat_map(accuracy_matrix)
