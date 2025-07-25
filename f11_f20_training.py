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
                # alpha=0.4
                # Update prototype for an existing pseudo-class
                self.prototypes[pseudo_label] = (
                    self.prototypes[pseudo_label] * self.class_counts[pseudo_label] + feature
                ) / (self.class_counts[pseudo_label] + 1)
                self.class_counts[pseudo_label] += 1  # Increment the count for the pseudo-class
                # self.prototypes[pseudo_label] = (
                # (1 - alpha) * self.prototypes[pseudo_label] + alpha * feature)

def save_model(model, path):
    """
    Save a model to the specified path using pickle.
    Args:
        model: The model object to save.
        path (str): The path to save the model file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Load a model from the specified path using pickle.
    Args:
        path (str): The path of the saved model file.
    Returns:
        Loaded model object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def task1_train_features():
    """
    Train the classifier incrementally on pre-extracted feature datasets (D11 to D20).
    Save each updated model to Google Drive after training and log the time taken for each step.
    """
    # Path to the pre-trained model on D1-D10
    model_path = 'models/f10.pkl'

    # Load the pre-trained classifier model
    classifier = load_model(model_path)
    
    # print("Loaded pre-trained model from", model_path)
    # Directory containing pre-extracted training features
    feature_dir = "part_two_dataset/eb3_extracted_features/train"
    timings = []

    # Loop through feature datasets D11 to D20
    for i in range(11, 21):
        start_time = time.time()  # Start timer for the current training step

        # Load the feature dataset for the current iteration
        feature_path = os.path.join(feature_dir, f"{i - 10}_train_features.npy")
        features = np.load(feature_path)
        # Note: Since labels are unavailable, pseudo-labels will be used for training

        # Generate pseudo-labels using the current state of the classifier
        pseudo_labels,confidences = classifier.predict(features)
        mask=confidences>0.8
        pseudo_labels=pseudo_labels[mask]
        features=features[mask]
        # Update the classifier incrementally with the new features and pseudo-labels
        classifier.update_classifier(features, pseudo_labels)

        # Save the updated classifier model to Google Drive
        save_path = f"models/f{i}.pkl"
        save_model(classifier, save_path)
        print(f"Trained and saved model f{i}")

        # Calculate and log the time taken for this iteration
        end_time = time.time()
        elapsed_time = end_time - start_time
        timings.append((i, elapsed_time))
        print(f"Time taken to train model f{i}: {elapsed_time:.2f} seconds")

    # Print summary of timings for all datasets
    print("\nTraining Timings:")
    for dataset, timing in timings:
        print(f"Dataset {dataset}: {timing:.2f} seconds")

task1_train_features()