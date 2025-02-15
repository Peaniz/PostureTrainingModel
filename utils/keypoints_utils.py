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

def extract_head_keypoints(image):
    """Extract head keypoints (nose, eyes, ears)"""
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # MediaPipe indices for head landmarks
        head_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in head_indices]).flatten()
        return head_points
    return None

def extract_body_keypoints(image):
    """Extract torso keypoints (shoulders, chest, hips)"""
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        body_indices = [11, 12, 23, 24]  # MediaPipe indices for body landmarks
        body_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in body_indices]).flatten()
        return body_points
    return None

def extract_arm_keypoints(image):
    """Extract arm keypoints (shoulders to wrists)"""
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        arm_indices = [11, 13, 15, 12, 14, 16]  # MediaPipe indices for arm landmarks
        arm_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in arm_indices]).flatten()
        return arm_points
    return None

def extract_leg_keypoints(image):
    """Extract leg keypoints (hips to ankles)"""
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        leg_indices = [23, 25, 27, 24, 26, 28]  # MediaPipe indices for leg landmarks
        leg_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in leg_indices]).flatten()
        return leg_points
    return None

def extract_foot_keypoints(image):
    """Extract foot keypoints (feet landmarks)"""
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        foot_indices = [29, 30, 31, 32]  # MediaPipe indices for foot landmarks
        foot_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in foot_indices]).flatten()
        return foot_points
    return None

def get_pose_results(image):
    """
    Get raw MediaPipe pose results for visualization.
    """
    results = pose.process(image)
    return results