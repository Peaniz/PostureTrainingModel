import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import time
import os
import json
import pickle
import matplotlib.pyplot as plt

def build_posture_model(input_shape, num_classes):
    """Build model for posture classification"""
    inputs = tf.keras.Input(shape=(input_shape,))
    
    # First block
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second block
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third block
    x = Dense(32, 
             activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_leg_model(input_shape):
    """Build model for leg position detection"""
    inputs = tf.keras.Input(shape=(input_shape,))
    
    # First block
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second block
    x = Dense(32, 
             activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer - binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_neck_model(input_shape, num_classes):
    """Build model for neck posture classification"""
    inputs = tf.keras.Input(shape=(input_shape,))
    
    # First block
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second block
    x = Dense(32, 
             activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def augment_data(X, y, noise_factor=0.05, num_augmentations=1):
    """Add noise to data for augmentation"""
    X_augmented = [X]
    y_augmented = [y]
    
    for _ in range(num_augmentations):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noise = X + noise
        X_augmented.append(X_noise)
        y_augmented.append(y)
    
    return np.vstack(X_augmented), np.hstack(y_augmented)

def plot_training_history(history, model_name):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower()}_training_history.png')
    plt.close()

def train_posture_model(X_train, y_train, X_test, y_test, num_classes):
    """Train and evaluate posture model"""
    print("\n=== Training Posture Model ===")
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    # Build model
    model = build_posture_model(X_train.shape[1], num_classes)
    
    # Define callbacks
    checkpoint_path = 'models/pose_classifier.h5'
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
            save_weights_only=False
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

def train_leg_model(X_train, y_train, X_test, y_test):
    """Train and evaluate leg model"""
    print("\n=== Training Leg Model ===")
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
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
            save_weights_only=False
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

def train_neck_model(X_train, y_train, X_test, y_test, num_classes):
    """Train and evaluate neck model"""
    print("\n=== Training Neck Model ===")
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    # Build model
    model = build_neck_model(X_train.shape[1], num_classes)
    
    # Define callbacks
    checkpoint_path = 'models/neck_classifier.h5'
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
            save_weights_only=False
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

def main():
    # Start timing
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    
    try:
        # Load posture data
        print("Loading posture data...")
        X_train_posture = np.load("data/splits/X_train_posture.npy", allow_pickle=True)
        X_test_posture = np.load("data/splits/X_test_posture.npy", allow_pickle=True)
        y_train_posture = np.load("data/splits/y_train_posture.npy", allow_pickle=True)
        y_test_posture = np.load("data/splits/y_test_posture.npy", allow_pickle=True)
        posture_classes = np.load("data/splits/posture_classes.npy", allow_pickle=True)
        
        # Load leg data
        print("\nLoading leg data...")
        X_train_leg = np.load("data/splits/X_train_leg.npy", allow_pickle=True)
        X_test_leg = np.load("data/splits/X_test_leg.npy", allow_pickle=True)
        y_train_leg = np.load("data/splits/y_train_leg.npy", allow_pickle=True)
        y_test_leg = np.load("data/splits/y_test_leg.npy", allow_pickle=True)
        leg_classes = np.load("data/splits/leg_classes.npy", allow_pickle=True)
        
        # Load neck data
        print("\nLoading neck data...")
        X_train_neck = np.load("data/splits/X_train_neck.npy", allow_pickle=True)
        X_test_neck = np.load("data/splits/X_test_neck.npy", allow_pickle=True)
        y_train_neck = np.load("data/splits/y_train_neck.npy", allow_pickle=True)
        y_test_neck = np.load("data/splits/y_test_neck.npy", allow_pickle=True)
        neck_classes = np.load("data/splits/neck_classes.npy", allow_pickle=True)
        
        # Normalize posture data
        scaler_posture = StandardScaler()
        X_train_posture = scaler_posture.fit_transform(X_train_posture)
        X_test_posture = scaler_posture.transform(X_test_posture)
        
        # Save posture scaler
        with open("models/scaler_posture.pkl", "wb") as f:
            pickle.dump(scaler_posture, f)
        
        # Train posture model
        posture_model, posture_history, posture_results = train_posture_model(
            X_train_posture, y_train_posture,
            X_test_posture, y_test_posture,
            len(posture_classes)
        )
        
        # Normalize leg data
        scaler_leg = StandardScaler()
        X_train_leg = scaler_leg.fit_transform(X_train_leg)
        X_test_leg = scaler_leg.transform(X_test_leg)
        
        # Save leg scaler
        with open("models/scaler_leg.pkl", "wb") as f:
            pickle.dump(scaler_leg, f)
        
        # Train leg model
        leg_model, leg_history, leg_results = train_leg_model(
            X_train_leg, y_train_leg,
            X_test_leg, y_test_leg
        )
        
        # Normalize neck data
        scaler_neck = StandardScaler()
        X_train_neck = scaler_neck.fit_transform(X_train_neck)
        X_test_neck = scaler_neck.transform(X_test_neck)
        
        # Save neck scaler
        with open("models/scaler_neck.pkl", "wb") as f:
            pickle.dump(scaler_neck, f)
        
        # Train neck model
        neck_model, neck_history, neck_results = train_neck_model(
            X_train_neck, y_train_neck,
            X_test_neck, y_test_neck,
            len(neck_classes)
        )
        
        # Plot training histories
        plot_training_history(posture_history, "Posture")
        plot_training_history(leg_history, "Leg")
        plot_training_history(neck_history, "Neck")
        
        # Verify models exist
        if not os.path.exists('models/pose_classifier.h5'):
            raise Exception("Posture model not saved correctly")
        if not os.path.exists('models/leg_classifier.h5'):
            raise Exception("Leg model not saved correctly")
        if not os.path.exists('models/neck_classifier.h5'):
            raise Exception("Neck model not saved correctly")
        
        # Create and save metadata
        metadata = {
            "posture_classes": posture_classes.tolist(),
            "leg_classes": leg_classes.tolist(),
            "neck_classes": neck_classes.tolist(),
            "training_time": time.time() - start_time,
            "posture_test_acc": float(posture_results[1]),
            "leg_test_acc": float(leg_results[1]),
            "neck_test_acc": float(neck_results[1]),
            "train_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        print("\n=== Training Complete ===")
        print(f"Total training time: {metadata['training_time']:.2f} seconds")
        print(f"Posture model test accuracy: {metadata['posture_test_acc']:.4f}")
        print(f"Leg model test accuracy: {metadata['leg_test_acc']:.4f}")
        print(f"Neck model test accuracy: {metadata['neck_test_acc']:.4f}")
        print("Models and metadata saved to 'models/' directory")
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please run data preprocessing first (python scripts/preprocess_data.py)")
        return
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your data and try again.")
        return

if __name__ == "__main__":
    main()