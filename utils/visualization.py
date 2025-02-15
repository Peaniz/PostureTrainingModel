import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_landmarks(image, keypoints=None):
    """
    Draw pose landmarks on the image.
    
    Args:
        image: Input image
        keypoints: Numpy array of keypoints or MediaPipe results object
    """
    # If keypoints is a numpy array, skip drawing
    if isinstance(keypoints, np.ndarray):
        return image
        
    # If keypoints is MediaPipe results, draw landmarks
    if hasattr(keypoints, 'pose_landmarks'):
        mp_drawing.draw_landmarks(
            image, 
            keypoints.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
    return image