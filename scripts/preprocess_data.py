import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import time
from collections import Counter

def clean_dataset(df):
    """Clean and validate the dataset"""
    print("Cleaning dataset...")
    
    # Remove rows with NaN values in features (not in label)
    initial_rows = len(df)
    df = df.dropna(subset=df.columns[:-1])  # Don't check label column
    nan_removed = initial_rows - len(df)
    if nan_removed > 0:
        print(f"Removed {nan_removed} rows with NaN values in features")
    
    # Ensure consistent number of columns
    expected_cols = 100  # 99 keypoints + 1 label
    if df.shape[1] > expected_cols:
        print(f"Trimming extra columns from {df.shape[1]} to {expected_cols}")
        df = df.iloc[:, :expected_cols]
    elif df.shape[1] < expected_cols:
        print(f"Warning: Found only {df.shape[1]} columns, expected {expected_cols}")
        # Add missing columns with 0 values
        for i in range(df.shape[1], expected_cols-1):
            df[f'kp_{i}'] = 0
    
    return df

def fix_csv_file(filepath):
    """Fix CSV file with inconsistent columns"""
    print("Attempting to fix CSV file...")
    
    try:
        # Read the file line by line
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Process each line to ensure consistent columns
        fixed_lines = []
        for i, line in enumerate(lines):
            values = line.strip().split(',')
            if len(values) > 100:
                # Trim extra columns
                values = values[:100]
            elif len(values) < 100:
                # Add empty values for missing columns
                values.extend([''] * (100 - len(values)))
            fixed_lines.append(','.join(values))
        
        # Write fixed data to temporary file
        temp_filepath = filepath + '.temp'
        with open(temp_filepath, 'w') as file:
            file.write('\n'.join(fixed_lines))
        
        return temp_filepath
    except Exception as e:
        print(f"Error fixing CSV file: {str(e)}")
        return None

def load_and_filter_data(dataset_type):
    """Load and filter dataset based on type"""
    try:
        if dataset_type == "posture":
            df = pd.read_csv("data/processed/posture_dataset.csv")
            # Kiểm tra số cột - 36 features (12 keypoints * 3) + 1 label
            expected_columns = 37
        elif dataset_type == "leg":
            df = pd.read_csv("data/processed/leg_dataset.csv")
            # Kiểm tra số cột - 30 features (10 keypoints * 3) + 1 label
            expected_columns = 31
        elif dataset_type == "neck":
            df = pd.read_csv("data/processed/neck_dataset.csv")
            # Kiểm tra số cột - 33 features (11 keypoints * 3) + 1 label
            expected_columns = 34
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        # Check dataset integrity
        if df.shape[1] != expected_columns:
            print(f"Warning: {dataset_type} dataset has {df.shape[1]} columns, expected {expected_columns}")
        
        # Get features and labels
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Print dataset info
        print(f"\n{dataset_type.capitalize()} dataset loaded: {df.shape[0]} samples")
        print(f"Labels distribution: {Counter(y)}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading {dataset_type} dataset: {str(e)}")
        return None, None

def clean_dataset(X, dataset_type="posture"):
    """Clean dataset based on type"""
    print(f"\nCleaning {dataset_type} dataset...")
    print(f"Initial shape: {X.shape}")
    
    # Replace NaN with 0
    if np.isnan(X).any():
        print("Found NaN values, replacing with 0")
        X = np.nan_to_num(X, 0)
    
    # Convert to float32 for better memory usage
    X = X.astype(np.float32)
    
    print(f"Final shape: {X.shape}")
    return X

def prepare_data():
    """Prepare both posture and leg datasets"""
    # Process posture dataset
    X_posture, y_posture = load_and_filter_data("posture")
    X_posture = clean_dataset(X_posture, "posture")
    
    # Process leg dataset
    X_leg, y_leg = load_and_filter_data("leg")
    X_leg = clean_dataset(X_leg, "leg")
    
    # Create label encoders
    posture_encoder = LabelEncoder()
    leg_encoder = LabelEncoder()
    
    # Encode labels
    y_posture_encoded = posture_encoder.fit_transform(y_posture)
    y_leg_encoded = leg_encoder.fit_transform(y_leg)
    
    # Split posture data
    X_train_posture, X_test_posture, y_train_posture, y_test_posture = train_test_split(
        X_posture, y_posture_encoded, test_size=0.2, random_state=42
    )
    
    # Split leg data
    X_train_leg, X_test_leg, y_train_leg, y_test_leg = train_test_split(
        X_leg, y_leg_encoded, test_size=0.2, random_state=42
    )
    
    # Create splits directory if it doesn't exist
    os.makedirs("data/splits", exist_ok=True)
    
    # Save posture data
    np.save("data/splits/X_train_posture.npy", X_train_posture)
    np.save("data/splits/X_test_posture.npy", X_test_posture)
    np.save("data/splits/y_train_posture.npy", y_train_posture)
    np.save("data/splits/y_test_posture.npy", y_test_posture)
    
    # Save leg data
    np.save("data/splits/X_train_leg.npy", X_train_leg)
    np.save("data/splits/X_test_leg.npy", X_test_leg)
    np.save("data/splits/y_train_leg.npy", y_train_leg)
    np.save("data/splits/y_test_leg.npy", y_test_leg)
    
    # Save encoders
    np.save("data/splits/posture_classes.npy", posture_encoder.classes_)
    np.save("data/splits/leg_classes.npy", leg_encoder.classes_)
    
    print("\nData preparation completed!")
    print("Posture dataset shapes:")
    print(f"X_train: {X_train_posture.shape}")
    print(f"X_test: {X_test_posture.shape}")
    print("\nLeg dataset shapes:")
    print(f"X_train: {X_train_leg.shape}")
    print(f"X_test: {X_test_leg.shape}")

def main():
    """Process all datasets and prepare for training"""
    start_time = time.time()
    
    try:
        os.makedirs("data/splits", exist_ok=True)
        
        # Process posture dataset
        print("=== Xử lý dữ liệu tư thế ===")
        X_posture, y_posture = load_and_filter_data("posture")
        if X_posture is not None:
            X_posture = clean_dataset(X_posture, "posture")
            
            # Create and fit label encoder for posture
            posture_encoder = LabelEncoder()
            y_posture_encoded = posture_encoder.fit_transform(y_posture)
            
            # Split posture data
            X_train_posture, X_test_posture, y_train_posture, y_test_posture = train_test_split(
                X_posture, y_posture_encoded, test_size=0.2, random_state=42, stratify=y_posture_encoded
            )
            
            # Save posture data
            np.save("data/splits/X_train_posture.npy", X_train_posture)
            np.save("data/splits/X_test_posture.npy", X_test_posture)
            np.save("data/splits/y_train_posture.npy", y_train_posture)
            np.save("data/splits/y_test_posture.npy", y_test_posture)
            np.save("data/splits/posture_classes.npy", posture_encoder.classes_)
            
            print("\nThống kê dữ liệu tư thế:")
            print(f"Số mẫu train: {len(X_train_posture)}")
            print(f"Số mẫu test: {len(X_test_posture)}")
            print(f"Số features: {X_posture.shape[1]}")
            print(f"Các lớp: {posture_encoder.classes_}")
        
        # Process leg dataset
        print("\n=== Xử lý dữ liệu vị trí chân ===")
        X_leg, y_leg = load_and_filter_data("leg")
        if X_leg is not None:
            X_leg = clean_dataset(X_leg, "leg")
            
            # Create and fit label encoder for leg
            leg_encoder = LabelEncoder()
            y_leg_encoded = leg_encoder.fit_transform(y_leg)
            
            # Split leg data
            X_train_leg, X_test_leg, y_train_leg, y_test_leg = train_test_split(
                X_leg, y_leg_encoded, test_size=0.2, random_state=42, stratify=y_leg_encoded
            )
            
            # Save leg data
            np.save("data/splits/X_train_leg.npy", X_train_leg)
            np.save("data/splits/X_test_leg.npy", X_test_leg)
            np.save("data/splits/y_train_leg.npy", y_train_leg)
            np.save("data/splits/y_test_leg.npy", y_test_leg)
            np.save("data/splits/leg_classes.npy", leg_encoder.classes_)
            
            print("\nThống kê dữ liệu vị trí chân:")
            print(f"Số mẫu train: {len(X_train_leg)}")
            print(f"Số mẫu test: {len(X_test_leg)}")
            print(f"Số features: {X_leg.shape[1]}")
            print(f"Các lớp: {leg_encoder.classes_}")
            
        # Process neck dataset
        print("\n=== Xử lý dữ liệu tư thế cổ ===")
        X_neck, y_neck = load_and_filter_data("neck")
        if X_neck is not None:
            X_neck = clean_dataset(X_neck, "neck")
            
            # Create and fit label encoder for neck
            neck_encoder = LabelEncoder()
            y_neck_encoded = neck_encoder.fit_transform(y_neck)
            
            # Split neck data
            X_train_neck, X_test_neck, y_train_neck, y_test_neck = train_test_split(
                X_neck, y_neck_encoded, test_size=0.2, random_state=42, stratify=y_neck_encoded
            )
            
            # Save neck data
            np.save("data/splits/X_train_neck.npy", X_train_neck)
            np.save("data/splits/X_test_neck.npy", X_test_neck)
            np.save("data/splits/y_train_neck.npy", y_train_neck)
            np.save("data/splits/y_test_neck.npy", y_test_neck)
            np.save("data/splits/neck_classes.npy", neck_encoder.classes_)
            
            print("\nThống kê dữ liệu tư thế cổ:")
            print(f"Số mẫu train: {len(X_train_neck)}")
            print(f"Số mẫu test: {len(X_test_neck)}")
            print(f"Số features: {X_neck.shape[1]}")
            print(f"Các lớp: {neck_encoder.classes_}")

        print(f"\nThời gian xử lý: {time.time() - start_time:.2f} giây")

    except FileNotFoundError as e:
        print(f"\nLỗi: {str(e)}")
        print("Vui lòng chạy thu thập dữ liệu (capture_data.py) trước.")
        return
    except Exception as e:
        print(f"\nLỗi không mong muốn: {str(e)}")
        print("Vui lòng kiểm tra lại dữ liệu và thử lại.")
        return

if __name__ == "__main__":
    main()