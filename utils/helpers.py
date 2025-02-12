import numpy as np
import csv

def save_to_csv(keypoints, label, filename="data/processed/dataset.csv"):
    """
    Save keypoints and labels to a CSV file.
    """
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([*keypoints, label])

def load_dataset(filename="data/processed/dataset.csv"):
    """
    Load dataset from a CSV file.
    """
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]  # Keypoints
    y = data[:, -1]   # Labels
    return X, y