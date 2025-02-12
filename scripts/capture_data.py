import cv2
import os
from utils.keypoints_utils import extract_keypoints
from utils.helpers import save_to_csv

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
pose_label = "correct_sitting"  # Change this label for different poses

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract keypoints
    keypoints = extract_keypoints(frame)
    if keypoints is not None:
        # Save keypoints to CSV
        save_to_csv(keypoints, pose_label)

    # Show frame
    cv2.imshow("Capture Pose Data", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()