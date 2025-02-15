import cv2
import os
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.helpers import save_to_csv, save_raw
from utils.visualization import draw_landmarks

def main():
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Define posture classes
    postures = [
        "good_sitting",
        "bad_sitting",
        "sitting_forward",
        "sitting_leanback",
        "sitting_left",
        "sitting_right"
    ]
    current_posture_idx = 0
    pose_label = postures[current_posture_idx]

    recording = False
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get pose results for visualization
        pose_results = get_pose_results(frame)
        # Extract keypoints for saving
        keypoints = extract_keypoints(frame)
        
        # Draw landmarks on frame
        frame = draw_landmarks(frame, pose_results)
            
        if keypoints is not None and recording:
            # Save keypoints to CSV
            save_to_csv(keypoints, pose_label)
            # Save raw frame
            save_raw(frame, pose_label, frame_count)
            frame_count += 1

        # Add recording status and current posture to frame
        status = "Recording: ON" if recording else "Recording: OFF"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Current Posture: {pose_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show instructions
        instructions = [
            "Press 'r' to start/stop recording",
            "Press 'n' to switch posture",
            "Press 'q' to quit"
        ]
        y_offset = 110
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        # Show frame
        cv2.imshow("Capture Pose Data", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            if recording:
                print(f"Started recording {pose_label}")
            else:
                print(f"Stopped recording {pose_label}")
        elif key == ord("n"):
            recording = False
            current_posture_idx = (current_posture_idx + 1) % len(postures)
            pose_label = postures[current_posture_idx]
            print(f"Switched to {pose_label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()