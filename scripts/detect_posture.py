import cv2
import numpy as np
import tensorflow as tf
import json
import pickle
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.visualization import draw_landmarks
import time

def load_models():
    """Load models and required files"""
    try:
        # Load models
        posture_model = tf.keras.models.load_model('models/pose_classifier.h5')
        leg_model = tf.keras.models.load_model('models/leg_classifier.h5')
        
        # Load scalers
        with open('models/scaler_posture.pkl', 'rb') as f:
            scaler_posture = pickle.load(f)
        with open('models/scaler_leg.pkl', 'rb') as f:
            scaler_leg = pickle.load(f)
        
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return posture_model, leg_model, scaler_posture, scaler_leg, metadata
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def extract_and_preprocess_keypoints(keypoints, scaler_posture, scaler_leg):
    """Extract and preprocess keypoints for both models"""
    # Extract upper body keypoints for posture
    upper_indices = list(range(0, 23))
    upper_keypoints = []
    for idx in upper_indices:
        start_idx = idx * 3
        upper_keypoints.extend(keypoints[start_idx:start_idx + 3])
    upper_keypoints = np.array(upper_keypoints).reshape(1, -1)
    
    # Extract leg keypoints
    leg_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    leg_keypoints = []
    for idx in leg_indices:
        start_idx = idx * 3
        leg_keypoints.extend(keypoints[start_idx:start_idx + 3])
    leg_keypoints = np.array(leg_keypoints).reshape(1, -1)
    
    # Normalize keypoints
    upper_keypoints_normalized = scaler_posture.transform(upper_keypoints)
    leg_keypoints_normalized = scaler_leg.transform(leg_keypoints)
    
    return upper_keypoints_normalized, leg_keypoints_normalized

def draw_results(frame, leg_pred, posture_pred, leg_classes, posture_classes):
    """Draw detection results on frame with probabilities"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Draw leg position result
    leg_prob = leg_pred[0][0]
    leg_text = f"{leg_classes[1]}: {leg_prob:.1%}" if leg_prob > 0.5 else f"{leg_classes[0]}: {(1-leg_prob):.1%}"
    leg_color = (0, 255, 0) if leg_prob <= 0.5 else (0, 0, 255)  # Green for correct, Red for wrong
    cv2.putText(frame, leg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, leg_color, 2)

    # Only show posture predictions if legs are in correct position
    if leg_prob <= 0.5:  # Correct leg position
        # Draw all posture probabilities
        posture_probs = posture_pred[0]
        max_prob_idx = np.argmax(posture_probs)
        
        # Draw probabilities for all postures
        y_offset = 70
        for i, (posture, prob) in enumerate(zip(posture_classes, posture_probs)):
            # Format text
            text = f"{posture}: {prob:.1%}"
            
            # Determine color and size based on whether this is the highest probability
            if i == max_prob_idx:
                color = (0, 255, 0)  # Green for highest probability
                font_scale = 1.0
                thickness = 2
            else:
                color = (200, 200, 200)  # Gray for others
                font_scale = 0.6
                thickness = 1
            
            # Draw text
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness)
            y_offset += 30
    else:
        # Show warning about incorrect leg position
        warning = "Please correct leg position first!"
        cv2.putText(frame, warning, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 255), 2)

    return frame

def main():
    # Load models and required files
    posture_model, leg_model, scaler_posture, scaler_leg, metadata = load_models()
    if None in [posture_model, leg_model, scaler_posture, scaler_leg, metadata]:
        print("Error: Could not load required files")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nBắt đầu phát hiện tư thế...")
    print("Nhấn 'q' để thoát")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get pose results and keypoints
        results = get_pose_results(frame)[0]
        keypoints = extract_keypoints(frame)
        
        if results and keypoints is not None:
            # Draw landmarks
            frame = draw_landmarks(frame, results)
            
            # Preprocess keypoints
            upper_keypoints, leg_keypoints = extract_and_preprocess_keypoints(
                keypoints, scaler_posture, scaler_leg
            )
            
            # Predict leg position first
            leg_pred = leg_model.predict(leg_keypoints, verbose=0)
            
            # Only predict posture if legs are in correct position
            if leg_pred[0][0] <= 0.5:
                posture_pred = posture_model.predict(upper_keypoints, verbose=0)
            else:
                posture_pred = None
            
            # Draw results
            frame = draw_results(frame, leg_pred, posture_pred, metadata['leg_classes'], metadata['posture_classes'])
        
        # Show frame
        cv2.imshow('Phát hiện tư thế', frame)
        
        # Handle key events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()