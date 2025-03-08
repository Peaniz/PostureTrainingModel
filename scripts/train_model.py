import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import shutil
import time
import datetime
from sklearn.preprocessing import LabelEncoder
import pickle

def cleanup_old_models():
    """Xóa các model và training data cũ"""
    print("\nCleaning up old models and training data...")
    
    # Xóa model cũ
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)
    
    # Xóa training plots cũ
    if os.path.exists("data/training_plots"):
        shutil.rmtree("data/training_plots")
    os.makedirs("data/training_plots", exist_ok=True)

def validate_data(X_train, y_train, X_test, y_test):
    """Validate data shapes and values"""
    print("\nValidating data shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Check for NaN values
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError("Input data contains NaN values")
    
    # Check for infinite values
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        raise ValueError("Input data contains infinite values")
    
    # Verify label consistency
    n_classes = len(np.unique(y_train))
    if n_classes != len(np.unique(y_test)):
        raise ValueError("Inconsistent number of classes between train and test sets")
    
    return n_classes

def build_optimized_model(input_shape, num_classes):
    """Xây dựng model tối ưu cho side view detection"""
    print(f"\nBuilding model with input shape {input_shape} and {num_classes} classes")
    
    model = models.Sequential([
        # Input layer with normalization
        layers.Input(shape=(99,)),
        layers.BatchNormalization(),
        
        # First dense block with residual connection
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second dense block
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense block
        layers.Dense(64, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history, save_path="data/training_plots"):
    """Plot and save training metrics"""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def main():
    start_time = time.time()
    
    # Cleanup old models and data
    cleanup_old_models()
    
    try:
        # Load and prepare data
        print("\nLoading training data...")
        X_train = np.load("data/splits/X_train.npy")
        X_test = np.load("data/splits/X_test.npy")
        y_train = np.load("data/splits/y_train.npy")
        y_test = np.load("data/splits/y_test.npy")
        
        # Create and fit label encoder
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Save label encoder
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        print("Label encoder saved to models/label_encoder.pkl")
        print("Classes:", label_encoder.classes_)
        
        # Normalize data
        X_train_mean = X_train.mean()
        X_train_std = X_train.std()
        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train_std
        
        # Save normalization parameters
        np.save("models/normalization_mean.npy", X_train_mean)
        np.save("models/normalization_std.npy", X_train_std)
        
        # Validate data
        num_classes = validate_data(X_train, y_train, X_test, y_test)
        
        # Print class distribution
        print("\nClass distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples")
        
        # Build and compile model
        model = build_optimized_model(X_train.shape[1], num_classes)
        
        # Warmup learning rate schedule
        initial_learning_rate = 0.0001  # Start with lower learning rate
        warmup_epochs = 5
        max_lr = 0.001
        min_lr = 1e-6
        
        def warmup_cosine_decay_schedule(epoch):
            if epoch < warmup_epochs:
                return initial_learning_rate + (max_lr - initial_learning_rate) * (epoch / warmup_epochs)
            else:
                decay_epochs = 150 - warmup_epochs
                epoch_in_decay = epoch - warmup_epochs
                cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
                return min_lr + (max_lr - min_lr) * cosine_decay
        
        # Compile with optimized settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        model.summary()

        # Add callbacks
        callbacks = [
            # Early stopping with more relaxed conditions
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # Monitor loss instead of accuracy
                patience=40,         # Much longer patience
                restore_best_weights=True,
                min_delta=0.0001,    # Smaller minimum improvement required
                mode='min',          # Monitor for minimizing loss
                verbose=1
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Custom learning rate schedule
            tf.keras.callbacks.LearningRateScheduler(
                warmup_cosine_decay_schedule,
                verbose=1
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]

        # Train model with adjusted parameters
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,          # Increased batch size for stability
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        # Evaluate and save results
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Save final model
        model.save('models/pose_classifier.h5')
        print("Model saved as 'models/pose_classifier.h5'")
        
        # Create training report
        training_report = {
            'final_train_acc': history.history['accuracy'][-1] * 100,
            'final_val_acc': history.history['val_accuracy'][-1] * 100,
            'final_test_acc': test_accuracy * 100,
            'best_val_acc': max(history.history['val_accuracy']) * 100,
            'best_train_acc': max(history.history['accuracy']) * 100,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_test_loss': test_loss,
            'training_time': (time.time() - start_time) / 60
        }
        
        # Format report text
        report_text = f"""
        \nTraining Results:
        ================
        Final Training Accuracy: {training_report['final_train_acc']:.2f}%
        Final Validation Accuracy: {training_report['final_val_acc']:.2f}%
        Final Test Accuracy: {training_report['final_test_acc']:.2f}%
        
        Best Validation Accuracy: {training_report['best_val_acc']:.2f}%
        Best Training Accuracy: {training_report['best_train_acc']:.2f}%
        
        Final Training Loss: {training_report['final_train_loss']:.4f}
        Final Validation Loss: {training_report['final_val_loss']:.4f}
        Final Test Loss: {training_report['final_test_loss']:.4f}
        
        Training Time: {training_report['training_time']:.2f} minutes
        """
        print(report_text)

        # Plot and save training history
        plot_training_history(history)
        
        # Save training report
        with open("data/training_plots/training_report.txt", "w") as f:
            f.write(report_text)
            
        return training_report  # Return dictionary instead of text

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()