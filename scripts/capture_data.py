import cv2
import os
from utils.keypoints_utils import extract_keypoints
from utils.helpers import save_to_csv, save_raw
from utils.visualization import draw_landmarks

def main():
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    pose_label = "bad_sitting"  # Change this label for different poses

    recording = False
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract keypoints
        keypoints = extract_keypoints(frame)
        if keypoints is not None and recording:
            # Save keypoints to CSV
            save_to_csv(keypoints, pose_label)
            # Save raw frame
            save_raw(frame, pose_label, frame_count)
            frame_count += 1
            # Draw landmarks on the frame

        # Add recording status to frame
        status = "Recording: ON" if recording else "Recording: OFF"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Capture Pose Data", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()