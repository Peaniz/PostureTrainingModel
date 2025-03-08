import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Sử dụng model phức tạp hơn
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a, b, c):
    """Tính góc giữa ba điểm"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_side_view_angles(landmarks):
    """Phân tích các góc quan trọng cho góc nhìn ngang"""
    # Lấy các điểm mốc quan trọng
    shoulder = [landmarks[11].x, landmarks[11].y]  # Vai phải
    hip = [landmarks[23].x, landmarks[23].y]       # Hông phải
    knee = [landmarks[25].x, landmarks[25].y]      # Đầu gối phải
    ankle = [landmarks[27].x, landmarks[27].y]     # Mắt cá chân phải
    ear = [landmarks[7].x, landmarks[7].y]         # Tai phải
    
    # Tính các góc quan trọng
    upper_body_angle = calculate_angle(ear, shoulder, hip)  # Góc thân trên
    lower_body_angle = calculate_angle(shoulder, hip, knee) # Góc thân dưới
    leg_angle = calculate_angle(hip, knee, ankle)          # Góc chân
    
    return {
        'upper_body_angle': upper_body_angle,
        'lower_body_angle': lower_body_angle,
        'leg_angle': leg_angle
    }

def analyze_side_posture(angles):
    """Phân tích tư thế dựa trên các góc"""
    upper_angle = angles['upper_body_angle']
    lower_angle = angles['lower_body_angle']
    
    # Phân tích tư thế dựa trên góc
    if 160 <= upper_angle <= 180:
        return "good_sitting_side"
    elif upper_angle < 160:
        if upper_angle < 140:
            return "bad_sitting_forward_side"
        else:
            return "bad_sitting_backward_side"
    elif upper_angle > 180:
        if upper_angle > 200:
            return "too_lean_back"
        else:
            return "too_lean_right"
    
    return "unknown"

def extract_keypoints(image):
    """Extract pose keypoints using MediaPipe Pose."""
    results = pose.process(image)
    if results.pose_landmarks:
        # Lấy tất cả keypoints
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return keypoints.flatten()  # Return flattened array
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
    """Get raw MediaPipe pose results for visualization."""
    results = pose.process(image)
    if results.pose_landmarks:
        # Tính toán các góc
        angles = get_side_view_angles(results.pose_landmarks.landmark)
        posture = analyze_side_posture(angles)
        return results, angles, posture
    return None, None, None

def extract_side_view_keypoints(image):
    """Extract keypoints optimized for side view analysis"""
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Các điểm quan trọng cho góc nhìn bên
        side_indices = [
            7, 8,     # Tai phải/trái
            11, 12,   # Vai phải/trái
            23, 24,   # Hông phải/trái
            25, 26,   # Đầu gối phải/trái
            27, 28,   # Mắt cá chân phải/trái
        ]
        
        # Lấy keypoints cho góc nhìn bên
        side_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] 
                               for i in side_indices]).flatten()
        
        # Tính toán các góc
        angles = get_side_view_angles(landmarks)
        angles_array = np.array([
            angles['upper_body_angle'],
            angles['lower_body_angle'],
            angles['leg_angle']
        ]) / 180.0
        
        # Kết hợp keypoints và góc
        return np.concatenate([side_points, angles_array])
    return None

def analyze_leg_position(keypoints):
    """Analyze leg position from keypoints"""
    # Extract leg keypoints
    left_knee = keypoints[25]
    right_knee = keypoints[26]
    left_ankle = keypoints[27]
    right_ankle = keypoints[28]
    
    # Tính toán góc và khoảng cách để xác định tư thế chân
    # ... (thêm logic phân tích tư thế chân)