import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import time
import json
import pickle
from sklearn.preprocessing import StandardScaler

def build_posture_model(input_shape, num_classes):
    """Build model for posture classification with regularization"""
    model = Sequential([
        # First block - extract basic features
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second block - learn patterns
        Dense(64, 
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third block - final features
        Dense(32, 
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer with high dropout
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_leg_model(input_shape):
    """Build model for leg position detection"""
    model = Sequential([
        # Simple architecture for binary classification
        Dense(32, 
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.02),
              input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, 
              activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def augment_data(X, y, noise_levels=[0.01, 0.02, 0.03]):
    """Augment data with multiple noise levels and random scaling"""
    X_augmented = [X]
    y_augmented = [y]
    
    for noise in noise_levels:
        # Add random noise
        X_noisy = X + np.random.normal(0, noise, X.shape)
        X_augmented.append(X_noisy)
        y_augmented.append(y)
        
        # Add scaled versions
        scale_factors = [0.95, 1.05]  # 5% scaling up and down
        for scale in scale_factors:
            X_scaled = X * scale
            X_augmented.append(X_scaled)
            y_augmented.append(y)
    
    return np.concatenate(X_augmented), np.concatenate(y_augmented)

def train_posture_model(X_train, y_train, X_test, y_test, num_classes):
    """Train and evaluate posture model with improved training strategy"""
    print("\n=== Training Posture Model ===")
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    # Build model
    model = build_posture_model(X_train.shape[1], num_classes)
    
    # Define callbacks with cross-validation monitoring
    checkpoint_path = 'models/pose_classifier.h5'
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            mode='min'
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=False  # Save entire model
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=0.00001,
            mode='min'
        )
    ]
    
    # Train with reduced epochs and class weights
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Load the best model
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    return model, history, results

def train_leg_model(X_train, y_train, X_test, y_test):
    """Train and evaluate leg model with improved training strategy"""
    print("\n=== Training Leg Model ===")
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_levels=[0.01, 0.02])
    
    # Build model
    model = build_leg_model(X_train.shape[1])
    
    # Define callbacks
    checkpoint_path = 'models/leg_classifier.h5'
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min'
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=False  # Save entire model
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            mode='min'
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.2,
        epochs=40,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Load the best model
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    return model, history, results

def plot_training_history(history, model_type):
    """Plot and save training history"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(f'{model_type} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'{model_type} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'models/{model_type.lower()}_training_history.png')
    plt.close()

def main():
    # Start timing
    start_time = time.time()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    try:
        # Load and preprocess posture data
        print("\nLoading posture data...")
        X_train_posture = np.load("data/splits/X_train_posture.npy", allow_pickle=True)
        X_test_posture = np.load("data/splits/X_test_posture.npy", allow_pickle=True)
        y_train_posture = np.load("data/splits/y_train_posture.npy", allow_pickle=True)
        y_test_posture = np.load("data/splits/y_test_posture.npy", allow_pickle=True)
        posture_classes = np.load("data/splits/posture_classes.npy", allow_pickle=True)
        
        # Load and preprocess leg data
        print("\nLoading leg data...")
        X_train_leg = np.load("data/splits/X_train_leg.npy", allow_pickle=True)
        X_test_leg = np.load("data/splits/X_test_leg.npy", allow_pickle=True)
        y_train_leg = np.load("data/splits/y_train_leg.npy", allow_pickle=True)
        y_test_leg = np.load("data/splits/y_test_leg.npy", allow_pickle=True)
        leg_classes = np.load("data/splits/leg_classes.npy", allow_pickle=True)
        
        # Normalize posture data
        scaler_posture = StandardScaler()
        X_train_posture = scaler_posture.fit_transform(X_train_posture)
        X_test_posture = scaler_posture.transform(X_test_posture)
        
        # Normalize leg data
        scaler_leg = StandardScaler()
        X_train_leg = scaler_leg.fit_transform(X_train_leg)
        X_test_leg = scaler_leg.transform(X_test_leg)
        
        # Save scalers
        with open("models/scaler_posture.pkl", "wb") as f:
            pickle.dump(scaler_posture, f)
        with open("models/scaler_leg.pkl", "wb") as f:
            pickle.dump(scaler_leg, f)
        
        # Train posture model
        posture_model, posture_history, posture_results = train_posture_model(
            X_train_posture, y_train_posture,
            X_test_posture, y_test_posture,
            len(posture_classes)
        )
        
        # Train leg model
        leg_model, leg_history, leg_results = train_leg_model(
            X_train_leg, y_train_leg,
            X_test_leg, y_test_leg
        )
        
        # Plot training histories
        plot_training_history(posture_history, "Posture")
        plot_training_history(leg_history, "Leg")
        
        # Verify models exist
        if not os.path.exists('models/pose_classifier.h5') or not os.path.exists('models/leg_classifier.h5'):
            raise Exception("Model files not saved correctly")
        
        # Save model metadata
        metadata = {
            'posture_classes': posture_classes.tolist(),
            'leg_classes': leg_classes.tolist(),
            'posture_input_shape': int(X_train_posture.shape[1]),  # Convert to int for JSON
            'leg_input_shape': int(X_train_leg.shape[1]),
            'posture_test_accuracy': float(posture_results[1]),
            'leg_test_accuracy': float(leg_results[1]),
            'training_time': float((time.time() - start_time) / 60),
            'training_completed': True  # Add flag to indicate successful training
        }
        
        # Save metadata
        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        print("\n=== Training Completed Successfully ===")
        print(f"Models saved to:")
        print("- models/pose_classifier.h5")
        print("- models/leg_classifier.h5")
        print("- models/model_metadata.json")
        
        # Print results
        print(f"\nTotal training time: {metadata['training_time']:.2f} minutes")
        
        print("\nPosture Model:")
        print(f"Test accuracy: {metadata['posture_test_accuracy']:.2%}")
        print(f"Number of classes: {len(posture_classes)}")
        print("Classes:", posture_classes)
        
        print("\nLeg Model:")
        print(f"Test accuracy: {metadata['leg_test_accuracy']:.2%}")
        print(f"Number of classes: {len(leg_classes)}")
        print("Classes:", leg_classes)

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        # Save error state in metadata
        error_metadata = {
            'training_completed': False,
            'error_message': str(e),
            'error_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open("models/model_metadata.json", "w") as f:
            json.dump(error_metadata, f, indent=4)
        return

if __name__ == "__main__":
    main()