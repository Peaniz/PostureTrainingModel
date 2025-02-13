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
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
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
    # Read data using pandas
    df = pd.read_csv(filename, header=None)

    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column (labels)
    return X, y