import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_keypoints(image):
    """
    Extract pose keypoints using MediaPipe Pose.
    """
    results = pose.process(image)
    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        return keypoints
    else:
        return None