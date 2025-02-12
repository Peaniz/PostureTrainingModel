import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(image, results):
    """
    Draw pose landmarks on the image.
    """
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    return image