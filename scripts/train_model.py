import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    # Load data
    X_train = np.load("data/splits/X_train.npy")
    X_test = np.load("data/splits/X_test.npy")
    y_train = np.load("data/splits/y_train.npy")
    y_test = np.load("data/splits/y_test.npy")

    # Reshape data for CNN input (33 landmarks, 3 coordinates)
    X_train = X_train.reshape((X_train.shape[0], 33, 3, 1))
    X_test = X_test.reshape((X_test.shape[0], 33, 3, 1))

    # Build CNN model
    model = models.Sequential([
        layers.Input(shape=(33, 3, 1)),
        layers.Conv2D(32, (3, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 1)),
        layers.Conv2D(64, (3, 1), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Save model
    model.save("models/pose_classifier.h5")

if __name__ == "__main__":
    main()