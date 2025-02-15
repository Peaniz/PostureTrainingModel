import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_path="data/training_plots"):
    """Plot and save training metrics"""
    os.makedirs(save_path, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def main():
    # Load data
    print("\nLoading training data...")
    X_train = np.load("data/splits/X_train.npy")
    X_test = np.load("data/splits/X_test.npy")
    y_train = np.load("data/splits/y_train.npy")
    y_test = np.load("data/splits/y_test.npy")
    
    print(f"\nDataset Information:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Input shape: {X_train[0].shape}")

    # Reshape data for CNN input
    X_train = X_train.reshape((X_train.shape[0], 33, 3, 1))
    X_test = X_test.reshape((X_test.shape[0], 33, 3, 1))

    # Build enhanced CNN model
    print("\nBuilding model...")
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(33, 3, 1)),
        
        # First Convolutional Block
        layers.Conv2D(64, (3, 1), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 1)),
        layers.Dropout(0.3),
        
        # Second Convolutional Block
        layers.Conv2D(128, (3, 1), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 1)),
        layers.Dropout(0.3),
        
        # Third Convolutional Block
        layers.Conv2D(256, (3, 1), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 1)),
        layers.Dropout(0.3),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])

    # Compile model with learning rate scheduling
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Add early stopping and model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train model with increased epochs
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=200,  # Increased epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Generate detailed report
    training_report = f"""
    \nTraining Results:
    ================
    Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%
    Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%
    Final Test Accuracy: {test_accuracy*100:.2f}%
    
    Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%
    Best Training Accuracy: {max(history.history['accuracy'])*100:.2f}%
    
    Final Training Loss: {history.history['loss'][-1]:.4f}
    Final Validation Loss: {history.history['val_loss'][-1]:.4f}
    Final Test Loss: {test_loss:.4f}
    """
    print(training_report)

    # Plot training history
    plot_training_history(history)

    # Save model
    print("\nSaving model...")
    model.save("models/pose_classifier.h5")
    
    # Save training report
    with open("data/training_plots/training_report.txt", "w") as f:
        f.write(training_report)

    return training_report

if __name__ == "__main__":
    main()