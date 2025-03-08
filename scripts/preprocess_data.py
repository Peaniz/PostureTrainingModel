import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import time

def clean_dataset(df):
    """Clean and validate the dataset"""
    print("Cleaning dataset...")
    
    # Remove rows with NaN values
    initial_rows = len(df)
    df = df.dropna()
    nan_removed = initial_rows - len(df)
    if nan_removed > 0:
        print(f"Removed {nan_removed} rows with NaN values")
    
    # Ensure consistent number of columns
    expected_cols = 100  # Expected number of columns (33 keypoints * 3 coordinates + 1 label)
    if df.shape[1] > expected_cols:
        print(f"Trimming extra columns from {df.shape[1]} to {expected_cols}")
        df = df.iloc[:, :expected_cols]
    elif df.shape[1] < expected_cols:
        print(f"Warning: Found only {df.shape[1]} columns, expected {expected_cols}")
        # Add missing columns with NaN values
        for i in range(df.shape[1], expected_cols-1):
            df[f'kp_{i}'] = np.nan
    
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

def load_and_filter_data(filepath, valid_postures):
    """Load and filter data in chunks to avoid memory issues"""
    print("\nLoading and filtering data...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    try:
        # Generate column names
        keypoint_cols = [f'kp_{i}_{ax}' for i in range(33) for ax in ['x', 'y', 'z']]
        columns = keypoint_cols + ['label']
        
        # First try to read normally
        try:
            df = pd.read_csv(filepath, names=columns, header=0)
            df = clean_dataset(df)
        except pd.errors.ParserError:
            print("Error reading CSV file. Attempting to fix format...")
            
            # Try reading with error handling
            try:
                df = pd.read_csv(filepath, 
                               names=columns,
                               header=0,
                               on_bad_lines='skip',  # New parameter name
                               warn_bad_lines=True)
                df = clean_dataset(df)
            except Exception:
                # If that fails, try fixing the file
                temp_filepath = fix_csv_file(filepath)
                if temp_filepath:
                    df = pd.read_csv(temp_filepath, names=columns, header=0)
                    os.remove(temp_filepath)  # Clean up temporary file
                else:
                    raise Exception("Could not fix file format")
        
        # Filter valid postures
        filtered_df = df[df.iloc[:, -1].isin(valid_postures)]
        
        if filtered_df.empty:
            raise ValueError("No valid data found after filtering!")
        
        print(f"Successfully loaded {len(filtered_df)} samples")
        return filtered_df
        
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

def main():
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Define valid postures - cập nhật theo capture_data.py
    valid_postures = [
        "good_sitting_side",          # Tư thế ngồi đúng nhìn từ bên
        "bad_sitting_forward_side",   # Cúi người về phía trước
        "bad_sitting_backward_side",  # Ngả người ra sau
        "too_lean_left_side",        # Nghiêng người sang trái
        "too_lean_right_side",       # Nghiêng người sang phải
        "legs_crossed"               # Ngồi vắt chéo chân
    ]

    try:
        # Load and filter data
        print("\nProcessing dataset...")
        df = load_and_filter_data("data/processed/dataset.csv", valid_postures)
        
        # Print data distribution
        print("\nData distribution:")
        total_samples = len(df)
        for label in valid_postures:
            count = len(df[df.iloc[:, -1] == label])
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{label}: {count} samples ({percentage:.1f}%)")

        # Split features and labels
        print("\nPreparing data for training...")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Split data with stratification
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )

        # Save data
        print("\nSaving processed data...")
        np.save("data/splits/X_train.npy", X_train)
        np.save("data/splits/X_test.npy", X_test)
        np.save("data/splits/y_train.npy", y_train)
        np.save("data/splits/y_test.npy", y_test)
        
        # Save label encoder
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        # Print summary
        print("\nPreprocessing completed successfully!")
        print(f"Total samples: {len(X)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes: {len(valid_postures)}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")

    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please run data collection (capture_data.py) first to generate the dataset.")
        return
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Make sure you have collected data for all the defined postures.")
        return
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your data collection and try again.")
        return

if __name__ == "__main__":
    main()