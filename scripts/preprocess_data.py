import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.helpers import load_dataset

# Load dataset
def main():
    X, y = load_dataset()

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save splits
    np.save("data/splits/X_train.npy", X_train)
    np.save("data/splits/X_test.npy", X_test)
    np.save("data/splits/y_train.npy", y_train)
    np.save("data/splits/y_test.npy", y_test)

    # Save label encoder
    import pickle
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    main()