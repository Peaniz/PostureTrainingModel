import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
X_train = np.load("data/splits/X_train.npy")
X_test = np.load("data/splits/X_test.npy")
y_train = np.load("data/splits/y_train.npy")
y_test = np.load("data/splits/y_test.npy")

# Reshape data for CNN input
X_train = X_train.reshape((X_train.shape[0], 33, 3, 1))
X_test = X_test.reshape((X_test.shape[0], 33, 3, 1))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(33, 3, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(np.unique(y_train)), activation="softmax")
])

# Compile and train
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("models/pose_classifier.h5")