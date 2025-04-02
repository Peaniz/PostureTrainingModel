import cv2
import numpy as np
import tensorflow as tf
import json
import pickle
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.visualization import draw_landmarks
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import datetime
from collections import defaultdict, deque
import os
from scipy.spatial.distance import cosine

def load_models():
    """Load trained models and metadata"""
    try:
        # Load models
        posture_model = tf.keras.models.load_model('models/pose_classifier.h5')
        leg_model = tf.keras.models.load_model('models/leg_classifier.h5')
        neck_model = tf.keras.models.load_model('models/neck_classifier.h5')
        
        # Load scalers
        with open('models/scaler_posture.pkl', 'rb') as f:
            scaler_posture = pickle.load(f)
        with open('models/scaler_leg.pkl', 'rb') as f:
            scaler_leg = pickle.load(f)
        with open('models/scaler_neck.pkl', 'rb') as f:
            scaler_neck = pickle.load(f)
        
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        posture_classes = metadata['posture_classes']
        leg_classes = metadata['leg_classes']
        neck_classes = metadata['neck_classes']
        
        return (posture_model, leg_model, neck_model, 
                scaler_posture, scaler_leg, scaler_neck,
                posture_classes, leg_classes, neck_classes)
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None

def extract_and_preprocess_keypoints(results):
    """Extract and preprocess keypoints for all models"""
    # Extract keypoints for leg model (10 keypoints * 3 coordinates = 30 features)
    leg_keypoints_idx = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # Hip to feet
    leg_keypoints = []
    for idx in leg_keypoints_idx:
        if results.pose_landmarks:
            point = results.pose_landmarks.landmark[idx]
            leg_keypoints.extend([point.x, point.y, point.z])
        else:
            leg_keypoints.extend([0, 0, 0])
    
    # Extract keypoints for neck model (11 keypoints * 3 coordinates = 33 features)
    neck_keypoints_idx = list(range(0, 11))  # Head and face points (0-10)
    neck_keypoints = []
    for idx in neck_keypoints_idx:
        if results.pose_landmarks:
            point = results.pose_landmarks.landmark[idx]
            neck_keypoints.extend([point.x, point.y, point.z])
        else:
            neck_keypoints.extend([0, 0, 0])
    
    # Extract keypoints for posture model (12 keypoints * 3 coordinates = 36 features)
    posture_keypoints_idx = list(range(11, 23))  # Shoulders to hips (11-22)
    posture_keypoints = []
    for idx in posture_keypoints_idx:
        if results.pose_landmarks:
            point = results.pose_landmarks.landmark[idx]
            posture_keypoints.extend([point.x, point.y, point.z])
        else:
            posture_keypoints.extend([0, 0, 0])
    
    # Reshape arrays to match model input shapes
    leg_keypoints = np.array(leg_keypoints).reshape(1, -1)  # Shape: (1, 30)
    neck_keypoints = np.array(neck_keypoints).reshape(1, -1)  # Shape: (1, 33)
    posture_keypoints = np.array(posture_keypoints).reshape(1, -1)  # Shape: (1, 36)
    
    return leg_keypoints, neck_keypoints, posture_keypoints

def draw_results(frame, leg_pred, neck_pred, posture_pred, leg_classes, neck_classes, posture_classes, is_wifi_camera=False):
    """Draw detection results on frame with probabilities"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Adjust font sizes based on camera type
    if is_wifi_camera:
        main_font_scale = 0.35  # Same as capture_data.py
        secondary_font_scale = 0.30
        main_thickness = 1
        secondary_thickness = 1
        line_spacing = 15
    else:
        main_font_scale = 0.40  # Same as capture_data.py
        secondary_font_scale = 0.35
        main_thickness = 1
        secondary_thickness = 1
        line_spacing = 20
    
    # Draw leg position result
    leg_prob = leg_pred[0][0]
    leg_text = f"{leg_classes[1]}: {leg_prob:.1%}" if leg_prob > 0.5 else f"{leg_classes[0]}: {(1-leg_prob):.1%}"
    leg_color = (0, 255, 0) if leg_prob <= 0.5 else (0, 0, 255)  # Green for correct, Red for wrong
    cv2.putText(frame, leg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, leg_color, main_thickness)

    # Only continue to posture if legs are correct
    if leg_prob <= 0.5:  # Correct leg position
        # Check posture
        posture_probs = posture_pred[0]
        max_posture_idx = np.argmax(posture_probs)
        posture_text = f"{posture_classes[max_posture_idx]}: {posture_probs[max_posture_idx]:.1%}"
        # Check if this is a "good" posture
        is_good_posture = posture_classes[max_posture_idx].startswith("good_")
        posture_color = (0, 255, 0) if is_good_posture else (0, 0, 255)  # Green for correct, Red for wrong
        
        cv2.putText(frame, posture_text, (10, 30 + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 
                   main_font_scale, posture_color, main_thickness)
        
        # Only evaluate neck if both legs and posture are correct
        if is_good_posture:
            # Draw neck position result
            neck_probs = neck_pred[0]
            max_neck_idx = np.argmax(neck_probs)
            neck_text = f"{neck_classes[max_neck_idx]}: {neck_probs[max_neck_idx]:.1%}"
            neck_color = (0, 255, 0) if max_neck_idx == 0 else (0, 0, 255)  # Green for correct, Red for wrong
            cv2.putText(frame, neck_text, (10, 30 + line_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 
                      main_font_scale, neck_color, main_thickness)
            
            # Show all posture probabilities with smaller font if neck is also correct
            if max_neck_idx == 0:  # Correct neck position
                # Draw posture probabilities
                y_offset = 30 + line_spacing * 3
                for i, (posture, prob) in enumerate(zip(posture_classes, posture_probs)):
                    # Format text
                    text = f"{posture}: {prob:.1%}"
                    
                    # Determine color and size based on whether this is the highest probability
                    if i == max_posture_idx:
                        color = (0, 255, 0) if posture.startswith("good_") else (0, 0, 255)
                        font_scale = main_font_scale
                        thickness = main_thickness
                    else:
                        color = (200, 200, 200)  # Gray for others
                        font_scale = secondary_font_scale
                        thickness = secondary_thickness
                    
                    # Draw text
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale, color, thickness)
                    y_offset += line_spacing
            else:
                # Show warning about incorrect neck position
                warning = "Vui long dieu chinh tu the co truoc!"
                cv2.putText(frame, warning, (10, 30 + line_spacing * 3), cv2.FONT_HERSHEY_SIMPLEX, 
                          main_font_scale, (0, 0, 255), main_thickness)
        else:
            # Show warning about incorrect posture
            warning = "Vui long dieu chinh tu the than truoc!"
            cv2.putText(frame, warning, (10, 30 + line_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 
                      main_font_scale, (0, 0, 255), main_thickness)
    else:
        # Show warning about incorrect leg position
        warning = "Vui long dieu chinh vi tri chan truoc!"
        cv2.putText(frame, warning, (10, 30 + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 
                   main_font_scale, (0, 0, 255), main_thickness)

    return frame

def choose_camera():
    """Cho phép người dùng lựa chọn giữa camera mặc định hoặc camera WiFi"""
    print("\n=== LỰA CHỌN CAMERA ===")
    print("1: Camera mặc định của máy tính")
    print("2: Camera WiFi (ESP32-CAM hoặc IP camera)")
    
    while True:
        try:
            choice = int(input("Nhập lựa chọn (1 hoặc 2): "))
            if choice in [1, 2]:
                break
            print("Vui lòng nhập 1 hoặc 2!")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ!")
    
    if choice == 1:
        # Sử dụng camera mặc định
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Đã kết nối với camera mặc định.")
    else:
        # Sử dụng camera WiFi
        default_url = "http://192.168.1.21/video"
        url = input(f"Nhập URL của camera WiFi (Enter để sử dụng {default_url}): ")
        if not url:
            url = default_url
        
        # Thử các đường dẫn phổ biến nếu kết nối thất bại
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            urls_to_try = [
                f"{url.split('/')[0]}//{url.split('/')[2]}/video",
                f"{url.split('/')[0]}//{url.split('/')[2]}/stream",
                f"{url.split('/')[0]}//{url.split('/')[2]}:81/stream",
                f"{url.split('/')[0]}//{url.split('/')[2]}/mjpeg"
            ]
            
            for try_url in urls_to_try:
                print(f"Thử kết nối với: {try_url}")
                cap = cv2.VideoCapture(try_url)
                if cap.isOpened():
                    url = try_url
                    print(f"Đã kết nối thành công với: {url}")
                    break
        
        if not cap.isOpened():
            print("Không thể kết nối với camera WiFi. Chuyển sang camera mặc định.")
            cap = cv2.VideoCapture(0)
        else:
            print(f"Đã kết nối với camera WiFi: {url}")
    
    # Kiểm tra kết nối camera
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra lại.")
        return None
    
    # Đọc frame đầu tiên để kiểm tra
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera. Vui lòng kiểm tra lại.")
        return None
    
    return cap

def main():
    # Load models
    models = load_models()
    if models is None:
        print("Could not load models. Please train models first.")
        return
    
    posture_model, leg_model, neck_model, scaler_posture, scaler_leg, scaler_neck, posture_classes, leg_classes, neck_classes = models
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize posture tracker
    tracker = PostureTracker(posture_classes, leg_classes, neck_classes)
    
    # Khởi tạo bộ nhớ cho các keypoints được xác nhận là đúng
    correct_keypoints = {
        'leg': [],
        'posture': [],
        'neck': []
    }
    
    # Thêm hệ thống tham chiếu
    reference_system = initialize_reference_system()
    
    # Cờ hiệu cho chế độ xác nhận
    confirmation_mode = False
    current_confirmation = None
    
    # Chọn và khởi tạo camera
    cap = choose_camera()
    if cap is None:
        print("Không thể khởi tạo camera. Thoát chương trình.")
        return
    
    # Lưu thông tin camera WiFi nếu đang sử dụng
    is_wifi_camera = False
    camera_url = ""
    if isinstance(cap, cv2.VideoCapture) and cap.isOpened():
        properties = [cap.get(prop) for prop in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS]]
        # Kiểm tra xem có phải camera WiFi không dựa trên các thuộc tính
        if properties[0] != 1280 or properties[1] != 720:
            is_wifi_camera = True
            print("Da phat hien camera WiFi - Dieu chinh kich thuoc chu.")
            try:
                camera_url = cap.get(cv2.CAP_PROP_POS_MSEC)
            except:
                camera_url = "http://192.168.1.21/video"  # URL mặc định
    
    print("\nDa khoi tao camera va model thanh cong.")
    print("Nhan 'q' de thoat chuong trinh.")
    print("Nhan 'v' de xem tong quan (overview).")
    print("Nhan 's' de luu thong ke va bieu do.")
    print("Nhan 'c' de xac nhan tu the hien tai la DUNG.")
    print("Nhan 'r' de su dung cac keypoints da xac nhan de cap nhat model.")
    
    # Biến đếm frame và thời gian để tính FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Tạo cửa sổ có thể điều chỉnh kích thước
    cv2.namedWindow('Posture Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Posture Detection', 1280, 720)
    
    # Flag to show visualization
    show_visualization = False
    current_viz = None
    
    # Lưu keypoints hiện tại
    current_leg_keypoints = None
    current_neck_keypoints = None
    current_posture_keypoints = None
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Khong doc duoc frame tu camera.")
            if is_wifi_camera:
                print("Dang thu ket noi lai camera WiFi...")
                time.sleep(2)  # Chờ 2 giây trước khi thử lại
                cap = reconnect_camera(camera_url)
                continue
            else:
                break
        
        # Tính FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Only process frames if not showing visualization
        if not show_visualization:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and get pose landmarks
            results = pose.process(rgb_frame)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract and preprocess keypoints
                leg_keypoints, neck_keypoints, posture_keypoints = extract_and_preprocess_keypoints(results)
                
                # Lưu lại keypoints hiện tại
                current_leg_keypoints = leg_keypoints.copy()
                current_neck_keypoints = neck_keypoints.copy()
                current_posture_keypoints = posture_keypoints.copy()
                
                # Normalize keypoints using appropriate scalers
                leg_keypoints_normalized = scaler_leg.transform(leg_keypoints)
                neck_keypoints_normalized = scaler_neck.transform(neck_keypoints)
                posture_keypoints_normalized = scaler_posture.transform(posture_keypoints)
                
                # Make predictions
                leg_pred = leg_model.predict(leg_keypoints_normalized, verbose=0)
                neck_pred = neck_model.predict(neck_keypoints_normalized, verbose=0)
                posture_pred = posture_model.predict(posture_keypoints_normalized, verbose=0)
                
                # Điều chỉnh dự đoán dựa trên tham chiếu
                if reference_system['leg']['reference'] is not None:
                    similarity = calculate_similarity(leg_keypoints[0], reference_system['leg']['reference'])
                    leg_pred = adjust_prediction(leg_pred, similarity, reference_system['leg']['threshold'])
                
                if reference_system['neck']['reference'] is not None:
                    similarity = calculate_similarity(neck_keypoints[0], reference_system['neck']['reference'])
                    neck_pred = adjust_prediction(neck_pred, similarity, reference_system['neck']['threshold'])
                    
                if reference_system['posture']['reference'] is not None:
                    similarity = calculate_similarity(posture_keypoints[0], reference_system['posture']['reference'])
                    posture_pred = adjust_prediction(posture_pred, similarity, reference_system['posture']['threshold'])
                
                # Update posture tracker
                tracker.update(leg_pred, neck_pred, posture_pred)
                
                # Draw results with appropriate font size based on camera type
                frame = draw_results(frame, leg_pred, neck_pred, posture_pred, 
                                   leg_classes, neck_classes, posture_classes, is_wifi_camera)
                
                # Nếu đang trong chế độ xác nhận, hiển thị thông báo
                if confirmation_mode:
                    confirm_text = f"Xác nhận: {current_confirmation} đúng? (Y/N)"
                    cv2.putText(frame, confirm_text, (10, frame.shape[0] - 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Hiển thị FPS - cũng điều chỉnh kích thước font
            fps_font_scale = 0.30 if is_wifi_camera else 0.35
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (0, 255, 0), 1)
            
            # Add tracking status text
            if tracker.current_posture:
                posture_text = f"Current: {tracker.current_posture}"
                cv2.putText(frame, posture_text, (frame.shape[1] - 300, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (0, 255, 0), 1)
            
            # Hiển thị số lượng keypoints đã xác nhận
            confirmed_text = f"Confirmed: Leg({len(correct_keypoints['leg'])}), Posture({len(correct_keypoints['posture'])}), Neck({len(correct_keypoints['neck'])})"
            cv2.putText(frame, confirmed_text, (10, frame.shape[0] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (255, 255, 0), 1)
            
            # Add help text at the bottom
            help_text = "c: confirm, r: retrain, v: viz, s: save stats"
            cv2.putText(frame, help_text, (frame.shape[1] // 2 - 150, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (255, 255, 255), 1)
            
            # Thêm hiển thị thông tin tham chiếu
            if any(reference_system[part]['reference'] is not None for part in ['leg', 'posture', 'neck']):
                ref_text = "References: "
                ref_text += "Leg " if reference_system['leg']['reference'] is not None else ""
                ref_text += "Posture " if reference_system['posture']['reference'] is not None else ""
                ref_text += "Neck" if reference_system['neck']['reference'] is not None else ""
                cv2.putText(frame, ref_text, (10, frame.shape[0] - 70),
                          cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (255, 255, 0), 1)
            
            # Display the resulting frame
            cv2.imshow('Posture Detection', frame)
        else:
            # When in visualization mode, display the current visualization
            if current_viz is None:
                # Generate visualization if not already done
                viz_fig = tracker.create_summary_dashboard()
                
                # Convert matplotlib figure to OpenCV image
                canvas = FigureCanvas(viz_fig)
                canvas.draw()
                viz_img = np.array(canvas.renderer.buffer_rgba())
                viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGBA2BGR)
                
                # Resize to match frame size
                viz_img = cv2.resize(viz_img, (frame.shape[1], frame.shape[0]))
                current_viz = viz_img
                
                # Close matplotlib figure to free memory
                plt.close(viz_fig)
            
            # Display the visualization
            cv2.imshow('Posture Detection', current_viz)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Break the loop on 'q' key press
        if key == ord('q'):
            break
        # Toggle visualization on 'v' key press
        elif key == ord('v'):
            show_visualization = not show_visualization
            if show_visualization:
                current_viz = None  # Reset visualization to generate a new one
                print("Showing visualization dashboard")
            else:
                print("Resuming posture detection")
        # Save visualizations and statistics on 's' key press
        elif key == ord('s'):
            os.makedirs("reports", exist_ok=True)
            dashboard_path = tracker.save_visualization()
            print(f"Statistics and visualizations saved to 'reports' directory")
            
            if not show_visualization:
                save_frame = frame.copy()
                save_text = "Statistics and visualizations saved!"
                cv2.putText(save_frame, save_text, (frame.shape[1]//2 - 200, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Posture Detection', save_frame)
                cv2.waitKey(1000)  # Show for 1 second
        
        # Enter confirmation mode on 'c' key press
        elif key == ord('c'):
            if results.pose_landmarks and not confirmation_mode:
                confirmation_mode = True
                current_confirmation = "leg"
                print("Entering confirmation mode. Confirm if leg position is correct (Y/N)")
            elif confirmation_mode:
                print("Already in confirmation mode. Please complete current confirmation.")
        
        # Handle Y/N responses in confirmation mode
        elif key == ord('y') and confirmation_mode:
            print(f"Confirmed: {current_confirmation} position is correct")
            
            # Save the confirmed keypoints to reference system
            if current_confirmation == "leg" and current_leg_keypoints is not None:
                reference_system['leg']['keypoints'].append(current_leg_keypoints[0])
                reference_system['leg']['reference'] = calculate_reference(reference_system['leg']['keypoints'])
                current_confirmation = "posture"
                print("Now confirm if posture is correct (Y/N)")
            elif current_confirmation == "posture" and current_posture_keypoints is not None:
                reference_system['posture']['keypoints'].append(current_posture_keypoints[0])
                reference_system['posture']['reference'] = calculate_reference(reference_system['posture']['keypoints'])
                current_confirmation = "neck"
                print("Now confirm if neck position is correct (Y/N)")
            elif current_confirmation == "neck" and current_neck_keypoints is not None:
                reference_system['neck']['keypoints'].append(current_neck_keypoints[0])
                reference_system['neck']['reference'] = calculate_reference(reference_system['neck']['keypoints'])
                confirmation_mode = False
                current_confirmation = None
                print("All positions confirmed and reference points updated.")
                
                # Hiển thị thông báo hoàn thành
                confirm_frame = frame.copy()
                confirm_text = "Reference points updated!"
                cv2.putText(confirm_frame, confirm_text, (frame.shape[1]//2 - 200, frame.shape[0]//2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Posture Detection', confirm_frame)
                cv2.waitKey(1000)
        
        elif key == ord('n') and confirmation_mode:
            print(f"Not confirmed: {current_confirmation} position is incorrect")
            
            if current_confirmation == "leg":
                current_confirmation = "posture"
                print("Now confirm if posture is correct (Y/N)")
            elif current_confirmation == "posture":
                current_confirmation = "neck"
                print("Now confirm if neck position is correct (Y/N)")
            elif current_confirmation == "neck":
                confirmation_mode = False
                current_confirmation = None
                print("Confirmation completed.")
        
        # Thêm phím 'r' để reset hệ thống tham chiếu
        elif key == ord('r'):
            reference_system = initialize_reference_system()
            print("Reference system reset.")
            
            # Hiển thị thông báo
            reset_frame = frame.copy()
            reset_text = "Reference system reset!"
            cv2.putText(reset_frame, reset_text, (frame.shape[1]//2 - 200, frame.shape[0]//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('Posture Detection', reset_frame)
            cv2.waitKey(1000)
    
    # Save final statistics on exit
    if tracker.total_tracked_time > 10:  # Only save if we've tracked for more than 10 seconds
        print("Saving final statistics...")
        tracker.save_visualization("reports/final")
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

def retrain_model(model, new_keypoints, new_labels, scaler):
    """Cập nhật mô hình với dữ liệu mới"""
    # Chuyển đổi dữ liệu mới thành numpy arrays
    X_new = np.array(new_keypoints)
    y_new = np.array(new_labels)
    
    # Chuẩn hóa dữ liệu mới
    X_new_scaled = scaler.transform(X_new)
    
    # Chuyển đổi sang định dạng one-hot nếu cần
    if len(model.layers[-1].output_shape) > 1 and model.layers[-1].output_shape[1] > 1:
        num_classes = model.layers[-1].output_shape[1]
        y_new_oh = tf.keras.utils.to_categorical(y_new, num_classes=num_classes)
    else:
        y_new_oh = y_new
    
    # Fine-tune mô hình với dữ liệu mới
    # Sử dụng learning rate thấp để tránh phá hủy kiến thức cũ
    initial_weights = model.get_weights()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fine-tune mô hình trong một số epoch nhỏ
    model.fit(X_new_scaled, y_new_oh, epochs=5, batch_size=max(1, len(X_new)//10), verbose=0)
    
    # Lưu mô hình mới
    # model.save('path_to_model.h5')  # Uncomment nếu muốn lưu mô hình mới
    
    return model

class PostureTracker:
    """Track posture data over time for visualization"""
    def __init__(self, posture_classes, leg_classes, neck_classes, history_length=3600):
        # Store class names
        self.posture_classes = posture_classes
        self.leg_classes = leg_classes
        self.neck_classes = neck_classes
        
        # Time tracking
        self.start_time = time.time()
        self.current_time = self.start_time
        self.history_length = history_length  # How many seconds of history to keep
        
        # Initialize counters for each class
        self.posture_times = {posture: 0 for posture in posture_classes}
        self.leg_times = {position: 0 for position in leg_classes}
        self.neck_times = {position: 0 for position in neck_classes}
        
        # For time series visualization (store data points every second)
        self.timestamps = deque(maxlen=history_length)
        self.posture_history = {posture: deque(maxlen=history_length) for posture in posture_classes}
        self.leg_history = {position: deque(maxlen=history_length) for position in leg_classes}
        self.neck_history = {position: deque(maxlen=history_length) for position in neck_classes}
        
        # Track correct postures
        self.correct_time = 0
        self.total_tracked_time = 0
        
        # Current state
        self.current_posture = None
        self.current_leg = None
        self.current_neck = None
        self.last_update = time.time()
    
    def update(self, leg_pred, neck_pred, posture_pred):
        """Update tracked posture data"""
        current_time = time.time()
        elapsed = current_time - self.last_update
        self.total_tracked_time += elapsed
        
        # Get current predictions
        leg_prob = leg_pred[0][0]
        current_leg = self.leg_classes[1] if leg_prob > 0.5 else self.leg_classes[0]
        
        # Start with evaluating leg position
        if leg_prob <= 0.5:  # Correct leg position
            # Now check posture
            posture_probs = posture_pred[0]
            max_posture_idx = np.argmax(posture_probs)
            current_posture = self.posture_classes[max_posture_idx]
            
            # Check if this is a "good" posture
            is_good_posture = current_posture.startswith("good_")
            
            # Only evaluate neck if posture is also good
            if is_good_posture:
                neck_probs = neck_pred[0]
                max_neck_idx = np.argmax(neck_probs)
                current_neck = self.neck_classes[max_neck_idx]
                
                # Check if neck position is also correct
                if max_neck_idx == 0:  # Correct neck position
                    # We have fully correct posture
                    self.correct_time += elapsed
                else:
                    # Neck is not correct
                    current_neck = self.neck_classes[max_neck_idx]
            else:
                # Posture is not good, don't evaluate neck
                current_neck = None
        else:
            # Leg position is incorrect
            current_posture = None
            current_neck = None
        
        # Update counters for the parts we evaluated
        self.leg_times[current_leg] += elapsed
        if current_posture:
            self.posture_times[current_posture] += elapsed
        if current_neck:
            self.neck_times[current_neck] += elapsed
        
        # Update current state
        self.current_posture = current_posture
        self.current_leg = current_leg
        self.current_neck = current_neck
        self.last_update = current_time
        
        # Add data point for time series (every second)
        if not self.timestamps or current_time - self.timestamps[-1] >= 1.0:
            self.timestamps.append(current_time)
            
            # Store leg values
            for position in self.leg_classes:
                value = 1 if position == current_leg else 0
                self.leg_history[position].append(value)
            
            # Store neck values - track as None if not evaluated
            for position in self.neck_classes:
                value = 1 if position == current_neck else 0
                if current_neck is None and position == self.neck_classes[0]:  # Default to neutral when not evaluated
                    value = 0
                self.neck_history[position].append(value)
            
            # Store posture values - track as None if not evaluated
            for posture in self.posture_classes:
                value = 1 if posture == current_posture else 0
                if current_posture is None and posture == self.posture_classes[0]:  # Default to neutral when not evaluated
                    value = 0
                self.posture_history[posture].append(value)
    
    def get_overall_stats(self):
        """Get overall statistics as a dictionary"""
        return {
            "total_time": self.total_tracked_time,
            "correct_time": self.correct_time,
            "correct_percentage": (self.correct_time / self.total_tracked_time * 100) if self.total_tracked_time > 0 else 0,
            "posture_times": self.posture_times,
            "leg_times": self.leg_times,
            "neck_times": self.neck_times
        }

    def create_pie_chart(self):
        """Create a pie chart showing time distribution for each posture"""
        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Posture pie chart
        posture_labels = []
        posture_values = []
        for posture, value in self.posture_times.items():
            if value > 0:
                posture_labels.append(posture)
                posture_values.append(value)
        
        if sum(posture_values) > 0:
            ax1.pie(posture_values, labels=posture_labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Posture Distribution')
        else:
            ax1.text(0.5, 0.5, 'No posture data', ha='center', va='center')
            ax1.axis('off')
        
        # Leg position pie chart
        leg_labels = []
        leg_values = []
        for position, value in self.leg_times.items():
            if value > 0:
                leg_labels.append(position)
                leg_values.append(value)
        
        if sum(leg_values) > 0:
            ax2.pie(leg_values, labels=leg_labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Leg Position Distribution')
        else:
            ax2.text(0.5, 0.5, 'No leg data', ha='center', va='center')
            ax2.axis('off')
        
        # Neck position pie chart
        neck_labels = []
        neck_values = []
        for position, value in self.neck_times.items():
            if value > 0:
                neck_labels.append(position)
                neck_values.append(value)
        
        if sum(neck_values) > 0:
            ax3.pie(neck_values, labels=neck_labels, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Neck Position Distribution')
        else:
            ax3.text(0.5, 0.5, 'No neck data', ha='center', va='center')
            ax3.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_time_bar_chart(self):
        """Create a bar chart showing time spent in each posture"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Posture bar chart
        posture_names = list(self.posture_times.keys())
        posture_times = list(self.posture_times.values())
        bars1 = ax1.bar(posture_names, posture_times, color='skyblue')
        ax1.set_title('Time Spent in Each Posture')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(range(len(posture_names)))
        ax1.set_xticklabels(posture_names, rotation=45, ha='right')
        
        # Add time labels above bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s',
                        ha='center', va='bottom')
        
        # Leg position bar chart
        leg_names = list(self.leg_times.keys())
        leg_times = list(self.leg_times.values())
        bars2 = ax2.bar(leg_names, leg_times, color='lightgreen')
        ax2.set_title('Time Spent in Each Leg Position')
        ax2.set_ylabel('Time (seconds)')
        
        # Add time labels above bars
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s',
                        ha='center', va='bottom')
        
        # Neck position bar chart
        neck_names = list(self.neck_times.keys())
        neck_times = list(self.neck_times.values())
        bars3 = ax3.bar(neck_names, neck_times, color='salmon')
        ax3.set_title('Time Spent in Each Neck Position')
        ax3.set_ylabel('Time (seconds)')
        
        # Add time labels above bars
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_posture_timeline(self):
        """Create a timeline showing posture changes over time"""
        if len(self.timestamps) < 2:
            # Not enough data for timeline
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Not enough data for timeline', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        # Convert timestamps to relative time (in minutes)
        start = self.timestamps[0]
        relative_times = [(t - start) / 60 for t in self.timestamps]  # Minutes
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot posture timeline
        good_postures = [p for p in self.posture_classes if p.startswith('good_')]
        bad_postures = [p for p in self.posture_classes if not p.startswith('good_')]
        
        # Plot good postures
        for posture in good_postures:
            if posture in self.posture_history:
                ax1.plot(relative_times, list(self.posture_history[posture]), 
                        label=posture, marker='.', linestyle='-', linewidth=2, alpha=0.7)
        
        # Plot bad postures
        for posture in bad_postures:
            if posture in self.posture_history:
                ax1.plot(relative_times, list(self.posture_history[posture]), 
                        label=posture, marker='.', linestyle='-', linewidth=2, alpha=0.7)
        
        ax1.set_title('Posture Timeline')
        ax1.set_ylabel('Active (1=Yes, 0=No)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot leg position timeline
        for position in self.leg_classes:
            if position in self.leg_history:
                ax2.plot(relative_times, list(self.leg_history[position]), 
                        label=position, marker='.', linestyle='-', linewidth=2, alpha=0.7)
        
        ax2.set_title('Leg Position Timeline')
        ax2.set_ylabel('Active (1=Yes, 0=No)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot neck position timeline
        for position in self.neck_classes:
            if position in self.neck_history:
                ax3.plot(relative_times, list(self.neck_history[position]), 
                        label=position, marker='.', linestyle='-', linewidth=2, alpha=0.7)
        
        ax3.set_title('Neck Position Timeline')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Active (1=Yes, 0=No)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self):
        """Create a comprehensive dashboard with all statistics"""
        fig = plt.figure(figsize=(12, 12))
        
        # Define grid for subplots
        gs = fig.add_gridspec(3, 2)
        
        # Create summary text area
        ax_summary = fig.add_subplot(gs[0, 0])
        ax_summary.axis('off')
        
        # Summary stats
        stats = self.get_overall_stats()
        total_minutes = stats['total_time'] / 60
        correct_minutes = stats['correct_time'] / 60
        
        summary_text = (
            f"POSTURE DETECTION SUMMARY\n"
            f"------------------------\n"
            f"Total tracking time: {total_minutes:.1f} minutes\n"
            f"Time in correct posture: {correct_minutes:.1f} minutes ({stats['correct_percentage']:.1f}%)\n"
            f"Current posture: {self.current_posture if self.current_posture else 'Unknown/Incorrect'}\n"
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        ax_summary.text(0.05, 0.95, summary_text, va='top', fontsize=10, 
                       family='monospace', transform=ax_summary.transAxes)
        
        # Add pie chart of time distribution by posture
        ax_pie = fig.add_subplot(gs[0, 1])
        
        # Prepare data for pie chart (good vs bad postures)
        good_time = sum(time for posture, time in self.posture_times.items() 
                       if posture.startswith('good_'))
        bad_time = sum(time for posture, time in self.posture_times.items() 
                     if not posture.startswith('good_'))
        
        if good_time + bad_time > 0:
            ax_pie.pie([good_time, bad_time], 
                      labels=['Good Posture', 'Bad Posture'],
                      colors=['lightgreen', 'salmon'],
                      autopct='%1.1f%%', startangle=90)
            ax_pie.set_title('Good vs Bad Posture')
        else:
            ax_pie.text(0.5, 0.5, 'No posture data yet', ha='center', va='center')
            ax_pie.axis('off')
        
        # Bar charts for time in each position
        ax_bars = fig.add_subplot(gs[1, :])
        
        # Combine all posture types in one chart
        all_labels = []
        all_times = []
        
        # Add leg positions
        for position, time_val in self.leg_times.items():
            if time_val > 0:
                all_labels.append(position)
                all_times.append(time_val)
        
        # Add posture types
        for posture, time_val in self.posture_times.items():
            if time_val > 0:
                all_labels.append(posture)
                all_times.append(time_val)
        
        # Add neck positions
        for position, time_val in self.neck_times.items():
            if time_val > 0:
                all_labels.append(position)
                all_times.append(time_val)
        
        # Colors based on category type
        colors = ['lightgreen', 'salmon'] * len(self.leg_classes) + \
                ['purple' if 'good' in label else 'red' for label in self.posture_classes] + \
                ['lightblue', 'orange'] * len(self.neck_classes)
        
        bars = ax_bars.bar(range(len(all_labels)), all_times, color=colors[:len(all_labels)])
        ax_bars.set_title('Time Spent in Each Position/Posture')
        ax_bars.set_ylabel('Time (seconds)')
        ax_bars.set_xticks(range(len(all_labels)))
        ax_bars.set_xticklabels(all_labels, rotation=45, ha='right')
        
        # Add time labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_bars.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # Timeline chart
        ax_timeline = fig.add_subplot(gs[2, :])
        
        if len(self.timestamps) >= 2:
            # Convert timestamps to minutes
            start = self.timestamps[0]
            relative_times = [(t - start) / 60 for t in self.timestamps]  # Minutes
            
            # Calculate a "posture score" over time (1 = perfect, 0 = bad)
            posture_scores = []
            # Get good postures list
            good_postures = [p for p in self.posture_classes if p.startswith('good_')]
            
            for i in range(len(self.timestamps)):
                # Start with checking leg position
                leg_correct = self.leg_history[self.leg_classes[0]][i] == 1
                
                if leg_correct:
                    # Check posture - is any "good" posture active?
                    good_active = False
                    for posture in good_postures:
                        if posture in self.posture_history and self.posture_history[posture][i] == 1:
                            good_active = True
                            break
                    
                    if good_active:
                        # Check neck position
                        neck_correct = self.neck_history[self.neck_classes[0]][i] == 1
                        score = 1.0 if neck_correct else 0.7  # Good with correct/incorrect neck
                    else:
                        score = 0.4  # Correct legs but bad posture
                else:
                    score = 0.1  # Incorrect legs
                
                posture_scores.append(score)
            
            # Plot the score
            ax_timeline.plot(relative_times, posture_scores, 'b-', linewidth=2)
            ax_timeline.fill_between(relative_times, 0, posture_scores, alpha=0.3, color='blue')
            
            # Add threshold lines
            ax_timeline.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent')
            ax_timeline.axhline(y=0.6, color='lightgreen', linestyle='--', alpha=0.7, label='Good')
            ax_timeline.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Needs Improvement')
            ax_timeline.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Poor')
            
            ax_timeline.set_title('Posture Quality Over Time')
            ax_timeline.set_xlabel('Time (minutes)')
            ax_timeline.set_ylabel('Posture Score')
            ax_timeline.set_ylim(0, 1.1)
            ax_timeline.legend()
            ax_timeline.grid(True, alpha=0.3)
        else:
            ax_timeline.text(0.5, 0.5, 'Not enough data for timeline', 
                           ha='center', va='center', fontsize=12)
            ax_timeline.axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, output_dir="reports"):
        """Save all visualizations to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save individual visualizations
        self.create_pie_chart().savefig(f"{output_dir}/pie_chart_{timestamp}.png")
        self.create_time_bar_chart().savefig(f"{output_dir}/bar_chart_{timestamp}.png")
        self.create_posture_timeline().savefig(f"{output_dir}/timeline_{timestamp}.png")
        self.create_summary_dashboard().savefig(f"{output_dir}/dashboard_{timestamp}.png")
        
        # Save stats as text
        stats = self.get_overall_stats()
        with open(f"{output_dir}/stats_{timestamp}.txt", "w") as f:
            f.write("POSTURE DETECTION STATISTICS\n")
            f.write("===========================\n\n")
            
            f.write(f"Total tracking time: {stats['total_time']:.1f} seconds\n")
            f.write(f"Time in correct posture: {stats['correct_time']:.1f} seconds\n")
            f.write(f"Percentage of time in correct posture: {stats['correct_percentage']:.1f}%\n\n")
            
            f.write("POSTURE BREAKDOWN\n")
            for posture, time_val in stats['posture_times'].items():
                f.write(f"  {posture}: {time_val:.1f} seconds\n")
            
            f.write("\nLEG POSITION BREAKDOWN\n")
            for position, time_val in stats['leg_times'].items():
                f.write(f"  {position}: {time_val:.1f} seconds\n")
            
            f.write("\nNECK POSITION BREAKDOWN\n")
            for position, time_val in stats['neck_times'].items():
                f.write(f"  {position}: {time_val:.1f} seconds\n")
        
        print(f"Visualizations and statistics saved to {output_dir}/")
        return f"{output_dir}/dashboard_{timestamp}.png"

def initialize_reference_system():
    """Khởi tạo hệ thống tham chiếu keypoints"""
    return {
        'leg': {'keypoints': [], 'reference': None, 'threshold': 0.15},
        'posture': {'keypoints': [], 'reference': None, 'threshold': 0.20},
        'neck': {'keypoints': [], 'reference': None, 'threshold': 0.15}
    }

def calculate_reference(keypoints_list):
    """Tính toán keypoint tham chiếu từ danh sách các keypoints đã xác nhận"""
    if not keypoints_list:
        return None
    
    # Chuyển đổi list thành array và tính trung bình
    keypoints_array = np.array(keypoints_list)
    reference = np.mean(keypoints_array, axis=0)
    return reference

def calculate_similarity(keypoints, reference):
    """Tính độ tương đồng giữa keypoints hiện tại và tham chiếu"""
    if reference is None:
        return 0.0
    
    # Sử dụng cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(keypoints, reference)
    return similarity

def adjust_prediction(original_pred, similarity, threshold):
    """Điều chỉnh dự đoán dựa trên độ tương đồng với tham chiếu"""
    # Nếu độ tương đồng vượt ngưỡng, ưu tiên điểm tham chiếu
    if similarity > threshold:
        # Mức độ tin cậy tỉ lệ với độ tương đồng
        confidence = min(0.95, (similarity - threshold) / (1 - threshold) * 0.7 + 0.25)
        
        # Đối với model nhị phân (leg, neck)
        if len(original_pred[0]) == 1:
            # Đây là model nhị phân, điều chỉnh để ưu tiên tư thế đúng (class 0)
            adjusted_pred = np.array([[1 - confidence]])
        else:
            # Đối với model đa lớp (posture), tạo distribution mới
            adjusted_pred = np.zeros_like(original_pred)
            # Đặt lớp "good_posture" (thường là lớp 0) có xác suất cao hơn
            adjusted_pred[0][0] = confidence
            # Phân bố phần còn lại cho các lớp khác
            remaining = 1 - confidence
            for i in range(1, len(adjusted_pred[0])):
                adjusted_pred[0][i] = remaining / (len(adjusted_pred[0]) - 1)
        
        # Trộn dự đoán gốc và điều chỉnh
        blend_factor = min(1.0, similarity * 2)  # Mức độ ưu tiên tham chiếu
        final_pred = original_pred * (1 - blend_factor) + adjusted_pred * blend_factor
        return final_pred
    
    # Nếu không, giữ nguyên dự đoán gốc
    return original_pred 