import cv2
import numpy as np
import tensorflow as tf
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.visualization import draw_landmarks
import pickle

def get_multiple_predictions(prediction, label_encoder, threshold=0.5):
    """Get all postures with confidence above threshold"""
    postures = []
    for idx, conf in enumerate(prediction[0]):
        if conf >= threshold:
            posture = label_encoder.inverse_transform([idx])[0]
            postures.append((posture, conf))
    return sorted(postures, key=lambda x: x[1], reverse=True)

def wrap_text(text, max_width, font_face, font_scale, thickness):
    """Wrap text to fit within max_width pixels"""
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        # Add word to current line
        current_line.append(word)
        
        # Check width with current word
        line_text = ' '.join(current_line)
        (line_width, _) = cv2.getTextSize(line_text, font_face, font_scale, thickness)[0]
        
        # If too wide, remove last word and start new line
        if line_width > max_width:
            current_line.pop()  # Remove last word
            if current_line:  # If there are words in current line
                lines.append(' '.join(current_line))
            current_line = [word]  # Start new line with the word that didn't fit
    
    # Add remaining words
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def main():
    # Load model and label encoder
    model = tf.keras.models.load_model("models/pose_classifier.h5")
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    # Print available classes
    print("Available classes:", label_encoder.classes_)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Double width (default is 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Double height (default is 480)

    # Define colors for different postures
    posture_colors = {
        "good_sitting": (0, 255, 0),     # Green
        "bad_sitting": (0, 0, 255),      # Red
        "bad_sitting_forward": (0, 165, 255), # Orange
        "bad_sitting_leanback": (0, 165, 255),# Orange
        "too_far_left": (0, 0, 255),      # Red
        "too_far_right": (0, 0, 255)      # Red
    }

    # Define display regions (adjusted for larger window)
    LEFT_MARGIN = 20
    RIGHT_MARGIN = 800  # Doubled from 400
    TOP_MARGIN = 50
    LINE_HEIGHT = 60    # Increased for better readability

    # Define text parameters
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    MAX_TEXT_WIDTH = 400  # Maximum width for recommendations text

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame if needed
        if frame.shape[1] < 1280:  # If width is less than 1280
            frame = cv2.resize(frame, (1280, 720))

        # Get pose results and keypoints
        pose_results = get_pose_results(frame)
        keypoints = extract_keypoints(frame)
        
        frame = draw_landmarks(frame, pose_results)

        if keypoints is not None:
            # Predict posture
            keypoints = np.expand_dims(keypoints, axis=0)
            keypoints = keypoints.reshape((1, 33, 3, 1))
            prediction = model.predict(keypoints, verbose=0)
            
            # Get multiple predictions above 50% threshold
            postures = get_multiple_predictions(prediction, label_encoder, threshold=0.5)

            if postures:
                # Display title for postures (left side)
                cv2.putText(frame, "Detected Postures:", 
                           (LEFT_MARGIN, TOP_MARGIN), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Display detected postures (left side)
                y_offset = TOP_MARGIN + LINE_HEIGHT
                for posture, confidence in postures:
                    color = posture_colors.get(posture, (255, 255, 255))
                    cv2.putText(frame, f"{posture}: {confidence:.1%}", 
                               (LEFT_MARGIN, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    y_offset += LINE_HEIGHT

                # Collect recommendations
                recommendations = set()
                for posture, _ in postures:
                    if posture == "bad_sitting":
                        recommendations.add("Straighten your back and adjust your position")
                    elif posture == "bad_sitting_forward":
                        recommendations.add("Sit back and straighten your back")
                    elif posture == "bad_sitting_leanback":
                        recommendations.add("Sit up and move closer to desk")
                    elif posture in ["too_far_left", "too_far_right"]:
                        recommendations.add("Center yourself with the screen")

                # Display title for recommendations (right side)
                cv2.putText(frame, "Recommendations:", 
                           (RIGHT_MARGIN, TOP_MARGIN), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Display recommendations with text wrapping
                y_offset = TOP_MARGIN + LINE_HEIGHT
                for recommendation in recommendations:
                    # Wrap recommendation text
                    wrapped_lines = wrap_text(
                        "- " + recommendation, 
                        MAX_TEXT_WIDTH,
                        FONT_FACE,
                        FONT_SCALE,
                        FONT_THICKNESS
                    )
                    
                    # Display each line
                    for line in wrapped_lines:
                        cv2.putText(frame, line,
                                  (RIGHT_MARGIN, y_offset),
                                  FONT_FACE,
                                  FONT_SCALE,
                                  (255, 255, 255),
                                  FONT_THICKNESS)
                        y_offset += int(LINE_HEIGHT * 0.8)  # Slightly reduced spacing for wrapped lines
                    
                    y_offset += int(LINE_HEIGHT * 0.2)  # Add extra space between recommendations
            else:
                # Show message when no posture has high enough confidence
                cv2.putText(frame, "No clear posture detected", 
                           (LEFT_MARGIN, TOP_MARGIN), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show frame in a named window with specific size
        cv2.namedWindow("Posture Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Posture Detection", 1280, 720)
        cv2.imshow("Posture Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()