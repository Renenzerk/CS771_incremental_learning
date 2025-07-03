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


class FeatureExtractor:
    def __init__(self):
        # Load EfficientNet-B3 model pre-trained on ImageNet
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        # Remove the final fully connected (FC) layer to get features instead of predictions
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()  # Set model to evaluation mode (disables training-specific behavior like dropout)

        # Define the transformation pipeline to prepare input images for EfficientNet-B3
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((300, 300)),  # Resize image to the input size required by EfficientNet-B3
            transforms.ToTensor(),  # Convert image to a normalized PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean for normalization
                std=[0.229, 0.224, 0.225]   # ImageNet std deviation for normalization
            ),
        ])

    def extract(self, images):
        """
        Extract feature representations for a batch of images.
        Args:
            images (numpy.ndarray): Raw images as numpy arrays.
        Returns:
            numpy.ndarray: Extracted feature vectors for all input images.
        """
        features = []
        with torch.no_grad():  # Disable gradient calculation 
            for img in images:
                # Apply preprocessing transformations
                img_transformed = self.transform(img)
                img_transformed = img_transformed.unsqueeze(0)  # Add batch dimension to the input tensor
                # Pass the transformed image through the model to extract features
                feature = self.model(img_transformed).squeeze().numpy()
                features.append(feature)  # Collect the feature vector

        # Return all features as a numpy array
        return np.array(features)


# Instantiate the FeatureExtractor for EfficientNet-B3
extractor = FeatureExtractor()

# Function to process datasets and extract features
def process_datasets(data_dir, prefix, num_datasets, output_dir):
    """
    Processes multiple datasets, extracts features, and saves them.
    Args:
        data_dir (str): Directory containing the datasets.
        prefix (str): Prefix for the dataset filenames (e.g., 'train' or 'eval').
        num_datasets (int): Number of datasets to process.
        output_dir (str): Directory to save the extracted features.
    """
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each dataset by its index
    for i in range(1, num_datasets + 1):
        # Construct the file path for the current dataset
        data_path = os.path.join(data_dir, f"{i}_{prefix}_data.tar.pth")
        # Load the dataset (assumes it's stored in PyTorch's .pth format)
        dataset = torch.load(data_path)
        images = dataset['data'] 

        print(f"Processing {prefix} dataset {i}...")
        # Extract features from the images using the feature extractor
        features = extractor.extract(images)
        # Construct the output file path and save the features as a .npy file
        # saving the features to further use it for training and testing the model
        output_path = os.path.join(output_dir, f"{i}_{prefix}_features.npy")
        np.save(output_path, features)
        print(f"Saved features for {prefix} dataset {i}")

# Paths to datasets
train_data_dir1 = 'dataset/dataset/part_one_dataset/train_data'
eval_data_dir1 = 'dataset/dataset/part_one_dataset/eval_data'

train_data_dir2 = 'dataset/dataset/part_two_dataset/train_data'
eval_data_dir2 = 'dataset/dataset/part_two_dataset/eval_data'

# Directories for storing extracted features
train_output_dir1 = 'eb3_extracted_features/train'
eval_output_dir1 = 'eb3_extracted_features/eval'

train_output_dir2 = 'eb3_extracted_features/train'
eval_output_dir2 = 'eb3_extracted_features/eval'

# Process the datasets and extract features

process_datasets(train_data_dir1, 'train', 10, train_output_dir1)
process_datasets(eval_data_dir1, 'eval', 10, eval_output_dir1)

process_datasets(train_data_dir2, 'train', 10, train_output_dir2)
process_datasets(eval_data_dir2, 'eval', 10, eval_output_dir2)