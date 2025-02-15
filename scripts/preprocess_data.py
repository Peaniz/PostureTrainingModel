import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import time

def load_and_filter_data(filepath, valid_postures):
    """Load and filter data in chunks to avoid memory issues"""
    print("\nLoading and filtering data...")
    
    # Read data in chunks
    chunk_size = 1000
    chunks = []
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Filter valid postures
        filtered_chunk = chunk[chunk.iloc[:, -1].isin(valid_postures)]
        if not filtered_chunk.empty:
            chunks.append(filtered_chunk)
        
    # Combine filtered chunks
    return pd.concat(chunks, axis=0) if chunks else pd.DataFrame()

def main():
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Define valid postures
    valid_postures = [
        "good_sitting",
        "bad_sitting",
        "bad_sitting_forward",
        "bad_sitting_leanback",
        "too_far_left",
        "too_far_right"
    ]

    # Load and filter data
    df = load_and_filter_data("data/processed/dataset.csv", valid_postures)
    
    if df.empty:
        print("No valid data found!")
        return

    # Print data distribution
    print("\nData distribution:")
    for label in valid_postures:
        count = len(df[df.iloc[:, -1] == label])
        print(f"{label}: {count} samples")

    # Split features and labels
    print("\nPreparing data for training...")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_postures)
    y = label_encoder.transform(y)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )

    # Save data
    print("\nSaving processed data...")
    try:
        np.save("data/splits/X_train.npy", X_train)
        np.save("data/splits/X_test.npy", X_test)
        np.save("data/splits/y_train.npy", y_train)
        np.save("data/splits/y_test.npy", y_test)
        
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return

    # Print summary
    print("\nPreprocessing completed!")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()