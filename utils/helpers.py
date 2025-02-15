"""Helper functions for data processing and file operations."""

import datetime
import os
from typing import Any, List, Tuple

import csv
import cv2
import numpy as np
import pandas as pd


def save_to_csv(keypoints: List[float], label: int, filename: str = "data/processed/dataset.csv") -> None:
    """
    Save keypoints and labels to a CSV file.

    Args:
        keypoints: List of keypoint coordinates
        label: Class label
        filename: Path to output CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check if file exists to write header
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write header if file is new
        if not file_exists:
            # Create header with feature names
            header = [f"feature_{i}" for i in range(len(keypoints))] + ["label"]
            writer.writerow(header)
        writer.writerow([*keypoints, label])


def save_raw(image: np.ndarray, label: Any, filename: str = "data/raw/") -> None:
    """
    Save raw images to a directory, organized by label.
    The images will be saved in: filename/label/image_timestamp.png

    Args:
        image: Numpy array of the image to save
        label: Class label (will be converted to string)
        filename: Base directory path where images will be saved
    """
    # Ensure filename is a string
    if not isinstance(filename, str):
        filename = "data/raw/"

    # Create main directory and label subdirectory
    label_dir = os.path.join(filename, str(label))
    os.makedirs(label_dir, exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(label_dir, f"{timestamp}.png")
    cv2.imwrite(filepath, image)


def load_dataset(filename: str = "data/processed/dataset.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from a CSV file.

    Args:
        filename: Path to input CSV file

    Returns:
        Tuple containing:
            - X: numpy array of keypoints
            - y: numpy array of encoded labels
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset file not found at {filename}\n"
            "Please follow these steps:\n"
            "1. Run: python setup.py\n"
            "2. Run: python main.py --mode capture\n"
            "3. Collect data for all posture classes"
        )
    
    # Read data using pandas
    df = pd.read_csv(filename, header=0)  # Changed to read with header

    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column (labels)
    return X, y