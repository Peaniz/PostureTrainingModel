import cv2
import numpy as np
import tensorflow as tf
from utils.keypoints_utils import extract_keypoints
from utils.visualization import draw_landmarks
import pickle

def main():
# Load model and label encoder
    model = tf.keras.models.load_model("models/pose_classifier.h5")
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract keypoints
        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            # Predict posture
            keypoints = np.expand_dims(keypoints, axis=0)
            keypoints = keypoints.reshape((1, 33, 3, 1))
            prediction = model.predict(keypoints)
            posture = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            # Display posture
            cv2.putText(frame, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()