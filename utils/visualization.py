import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_landmarks(image, results=None):
    """
    Draw pose landmarks on the image.
    
    Args:
        image: Input image
        results: MediaPipe results object
    """
    annotated_image = image.copy()
    
    if results and hasattr(results, 'pose_landmarks'):
        # Customize drawing style
        landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green color for landmarks
            thickness=2,
            circle_radius=2
        )
        connection_drawing_spec = mp_drawing.DrawingSpec(
            color=(255, 255, 255),  # White color for connections
            thickness=2
        )
        
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec,
            connection_drawing_spec
        )
    
    return annotated_image