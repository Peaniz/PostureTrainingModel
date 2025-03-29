import cv2
import os
import numpy as np
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.helpers import save_to_csv, save_raw
from utils.visualization import draw_landmarks
import csv
import time
import shutil
import mediapipe as mp

def extract_leg_keypoints(keypoints):
    """Chỉ lấy các keypoint của chân"""
    # Các keypoint của chân trong MediaPipe Pose:
    # 23, 24: Hip (Left/Right)
    # 25, 26: Knee (Left/Right)
    # 27, 28: Ankle (Left/Right)
    # 29, 30: Heel (Left/Right)
    # 31, 32: Foot Index (Left/Right)
    leg_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    leg_keypoints = []
    
    for idx in leg_indices:
        start_idx = idx * 3
        leg_keypoints.extend(keypoints[start_idx:start_idx + 3])
    
    return np.array(leg_keypoints)

def extract_upper_body_keypoints(keypoints):
    """Chỉ lấy các keypoint phần thân trên (không bao gồm đầu và chân)"""
    # Sử dụng các keypoint từ 11-22: vai đến hông
    upper_indices = list(range(11, 23))  # 11-22: Từ vai đến hông
    upper_keypoints = []
    
    for idx in upper_indices:
        start_idx = idx * 3
        upper_keypoints.extend(keypoints[start_idx:start_idx + 3])
    
    return np.array(upper_keypoints)

def extract_neck_keypoints(keypoints):
    """Chỉ lấy các keypoint của đầu và cổ"""
    # Các keypoint của đầu và cổ trong MediaPipe Pose:
    # 0: Nose
    # 1-4: Left/Right eye inner/outer
    # 5-6: Left/Right ear
    # 7-10: Mouth left/right, left/right
    neck_indices = list(range(0, 11))  # 0-10: đầu, mặt, tai
    neck_keypoints = []
    
    for idx in neck_indices:
        start_idx = idx * 3
        neck_keypoints.extend(keypoints[start_idx:start_idx + 3])
    
    return np.array(neck_keypoints)

def save_to_csv(keypoints, label, is_leg_data=False, is_neck_data=False, csv_file=None):
    """Save keypoints to CSV file with proper formatting"""
    if csv_file is None:
        if is_leg_data:
            csv_file = "data/processed/leg_dataset.csv"
        elif is_neck_data:
            csv_file = "data/processed/neck_dataset.csv"
        else:
            csv_file = "data/processed/posture_dataset.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Process keypoints based on data type
    if is_leg_data:
        keypoints = extract_leg_keypoints(keypoints)
    elif is_neck_data:
        keypoints = extract_neck_keypoints(keypoints)
    else:
        keypoints = extract_upper_body_keypoints(keypoints)
        
    # Ensure keypoints is a flat array
    flat_keypoints = keypoints.flatten()
    
    # Create row with keypoints and label
    row = np.append(flat_keypoints, label)
    
    # Check if file exists to write header
    file_exists = os.path.isfile(csv_file)
    
    # Open file in append mode
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            # Create header with keypoint names and label
            if is_leg_data:
                header = [f'leg_kp_{i}_{ax}' for i in range(10) for ax in ['x', 'y', 'z']]
            elif is_neck_data:
                header = [f'neck_kp_{i}_{ax}' for i in range(11) for ax in ['x', 'y', 'z']]
            else:
                header = [f'kp_{i}_{ax}' for i in range(12) for ax in ['x', 'y', 'z']]
            header.append('label')
            writer.writerow(header)
        
        # Write data row
        writer.writerow(row)

def save_raw(frame, label, frame_count, raw_dir="data/raw"):
    """Save raw frame with timestamp"""
    # Create directory for this label if it doesn't exist
    label_dir = os.path.join(raw_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Save frame with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{label}_{timestamp}_{frame_count}.jpg"
    cv2.imwrite(os.path.join(label_dir, filename), frame)

def draw_controls_info(frame, recording, is_wifi_camera=False):
    """Hiển thị các điều khiển của phần mềm lên frame"""
    height, width = frame.shape[:2]
    
    # Điều chỉnh cỡ chữ tùy thuộc vào loại camera
    if is_wifi_camera:
        font_scale = 0.35
        thickness = 1
        line_spacing = 15
    else:
        font_scale = 0.40
        thickness = 1
        line_spacing = 20
    
    # Vẽ background mờ cho phần điều khiển
    controls_bg = np.zeros((150, 280, 3), dtype=np.uint8)
    alpha = 0.7
    
    roi = frame[height-150:height, width-280:width]
    cv2.addWeighted(controls_bg, alpha, roi, 1 - alpha, 0, roi)
    
    # Hiển thị thông tin điều khiển
    controls = [
        ("r", "Start/Stop recording"),
        ("n", "Next pose/position"),
        ("q", "Quit program")
    ]
    
    for i, (key, desc) in enumerate(controls):
        y_pos = height - 130 + i * line_spacing
        cv2.putText(frame, f"{key}: {desc}", (width - 270, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Hiển thị trạng thái recording
    rec_status = "RECORDING ON" if recording else "RECORDING OFF"
    rec_color = (0, 255, 0) if recording else (0, 0, 255)
    cv2.putText(frame, rec_status, (width - 270, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, rec_color, thickness)
    
    return frame

def draw_lean_guide(frame, posture_label, lean_angle=0, is_wifi_camera=False):
    """Vẽ trợ giúp góc nghiêng khi thu thập dữ liệu"""
    height, width = frame.shape[:2]
    
    # Điều chỉnh cỡ chữ dựa trên loại camera
    if is_wifi_camera:
        font_scale = 0.35
        thickness = 1
    else:
        font_scale = 0.40
        thickness = 1
    
    # Chỉ hiển thị khi thu thập dữ liệu tư thế nghiêng hoặc nếu góc nghiêng > 5 độ
    show_guide = "too_lean" in posture_label or abs(lean_angle) > 5
    if not show_guide:
        return frame
    
    # Vẽ trợ giúp hướng nghiêng
    center_x = width - 150
    center_y = height - 100
    radius = 35 if is_wifi_camera else 40  # Điều chỉnh kích thước đồng hồ với camera WiFi
    
    # Vẽ đường tròn và trục
    cv2.circle(frame, (center_x, center_y), radius, (200, 200, 200), thickness)
    cv2.line(frame, (center_x - radius, center_y), (center_x + radius, center_y), (200, 200, 200), 1)
    cv2.line(frame, (center_x, center_y - radius), (center_x, center_y + radius), (200, 200, 200), 1)
    
    # Vẽ vùng mục tiêu của góc nghiêng (15-30 độ)
    if "too_lean_left" in posture_label:
        # Nghiêng trái - vẽ khu vực màu xanh bên trái
        start_angle = 105  # 90 + 15 độ
        end_angle = 120    # 90 + 30 độ
        color = (255, 0, 0)  # Màu xanh dương
        text = "Nghieng TRAI"
    elif "too_lean_right" in posture_label:
        # Nghiêng phải - vẽ khu vực màu xanh bên phải
        start_angle = 60   # 90 - 30 độ
        end_angle = 75     # 90 - 15 độ
        color = (255, 0, 0)  # Màu xanh dương
        text = "Nghieng PHAI"
    else:
        # Nếu không phải thu thập dữ liệu nghiêng, chỉ hiển thị thông tin
        color = (200, 200, 200)
        text = "Goc nghieng"
    
    # Vẽ cung chỉ dẫn nếu đang thu thập dữ liệu nghiêng
    if "too_lean" in posture_label:
        axes = (radius, radius)
        angle = 0
        cv2.ellipse(frame, (center_x, center_y), axes, angle, start_angle, end_angle, color, 2)
    
    # Vẽ kim chỉ góc nghiêng thực tế
    angle_rad = np.radians(lean_angle)
    end_x = center_x + int(radius * np.sin(angle_rad))
    end_y = center_y - int(radius * np.cos(angle_rad))
    
    # Màu kim phụ thuộc vào góc nghiêng
    needle_color = (0, 255, 0)  # Xanh lá khi thẳng
    if lean_angle < -15:
        needle_color = (0, 0, 255)  # Đỏ khi nghiêng trái nhiều
    elif lean_angle > 15:
        needle_color = (0, 0, 255)  # Đỏ khi nghiêng phải nhiều
    
    cv2.line(frame, (center_x, center_y), (end_x, end_y), needle_color, 2)
    
    # Hiển thị góc nghiêng bằng số
    angle_text = f"Goc nghieng: {lean_angle:.1f}°"
    y_offset = radius + 15 if is_wifi_camera else radius + 20
    cv2.putText(frame, angle_text, (center_x - (radius + 40), center_y + y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Thêm chú thích
    y_offset = center_y + radius + 15
    cv2.putText(frame, text, (center_x - 50, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Thêm hướng dẫn nếu đang thu thập dữ liệu nghiêng
    if "too_lean" in posture_label:
        y_offset += 15 if is_wifi_camera else 20
        cv2.putText(frame, "Giu nghieng tu 15-30°", (center_x - 70, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), 1)
    
    return frame

def calculate_lean_angle(results):
    """Tính góc nghiêng của cơ thể dựa trên vai và hông"""
    if not results or not results.pose_landmarks:
        return 0
    
    # Lấy tọa độ vai
    left_shoulder = results.pose_landmarks.landmark[11]  # Left shoulder
    right_shoulder = results.pose_landmarks.landmark[12]  # Right shoulder
    
    # Lấy tọa độ hông
    left_hip = results.pose_landmarks.landmark[23]  # Left hip
    right_hip = results.pose_landmarks.landmark[24]  # Right hip
    
    # Tính điểm trung tâm vai và hông
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_hip_x = (left_hip.x + right_hip.x) / 2
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    mid_hip_y = (left_hip.y + right_hip.y) / 2
    
    # Tính góc nghiêng (độ)
    # Góc âm = nghiêng trái, góc dương = nghiêng phải
    if mid_hip_y - mid_shoulder_y == 0:  # Tránh chia cho 0
        return 0
    
    # Tính góc so với trục dọc (độ)
    angle_rad = np.arctan((mid_shoulder_x - mid_hip_x) / (mid_hip_y - mid_shoulder_y))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

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
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Define posture classes (không bao gồm tư thế chân)
    postures = [
        "good_sitting_side",          # 0: Ngồi thẳng lưng
        "bad_sitting_backward_side",  # 1: Ngồi ngả ra sau
        "bad_sitting_forward_side",   # 2: Cúi người về phía trước
        "too_lean_left_side",        # 3: Nghiêng người sang trái
        "too_lean_right_side",       # 4: Nghiêng người sang phải
    ]
    
    # Define leg positions - chỉ có 2 trạng thái
    leg_positions = [
        "leg_right_position",        # 0: Chân đặt đúng tư thế (song song, bàn chân sát sàn)
        "leg_wrong_position"         # 1: Chân đặt sai tư thế (vắt chéo, gác lên ghế, duỗi thẳng...)
    ]
    
    # Define neck positions - 2 trạng thái
    neck_positions = [
        "neck_right_position",       # 0: Cổ thẳng, đầu không cúi/ngả
        "neck_wrong_position"        # 1: Cổ cúi, đầu gập xuống hoặc ngả
    ]
    
    # Mode selection
    print("\nChon che do thu thap du lieu:")
    print("1: Thu thap du lieu tu the")
    print("2: Thu thap du lieu vi tri chan")
    print("3: Thu thap du lieu tu the co")
    while True:
        try:
            mode = int(input("Nhap lua chon (1, 2 hoac 3): "))
            if mode in [1, 2, 3]:
                break
            print("Vui long nhap 1, 2 hoac 3!")
        except ValueError:
            print("Vui long nhap mot so hop le!")

    # Set up based on mode
    if mode == 1:
        current_labels = postures
        is_leg_data = False
        is_neck_data = False
        data_type = "tu the"
        dataset_file = "data/processed/posture_dataset.csv"
    elif mode == 2:
        current_labels = leg_positions
        is_leg_data = True
        is_neck_data = False
        data_type = "vi tri chan"
        dataset_file = "data/processed/leg_dataset.csv"
    else:  # mode == 3
        current_labels = neck_positions
        is_leg_data = False
        is_neck_data = True
        data_type = "tu the co"
        dataset_file = "data/processed/neck_dataset.csv"
    
    # Handle existing dataset file
    if os.path.exists(dataset_file):
        print(f"\nDa tim thay file du lieu {data_type} cu.")
        print("Ban muon xu ly file cu nhu the nao?")
        print("1: Ghi de (xoa du lieu cu)")
        print("2: Tao backup va bat dau file moi")
        print("3: Giu nguyen va them du lieu vao file cu")
        
        while True:
            try:
                file_choice = int(input("Nhap lua chon cua ban (1-3): "))
                if file_choice in [1, 2, 3]:
                    break
                print("Vui long nhap 1, 2 hoac 3!")
            except ValueError:
                print("Vui long nhap mot so hop le!")
        
        if file_choice == 1:
            os.remove(dataset_file)
            print(f"Da xoa file du lieu {data_type} cu")
        elif file_choice == 2:
            # Create backup directory if it doesn't exist
            backup_dir = "data/backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_file = os.path.join(backup_dir, f"{os.path.basename(dataset_file)}.{timestamp}.backup")
            
            # Copy current file to backup
            shutil.copy2(dataset_file, backup_file)
            
            # Remove current file to start fresh
            os.remove(dataset_file)
            print(f"Da sao luu file cu vao: {backup_file}")
            print(f"Da tao file moi de ghi du lieu")
        else:
            print(f"Se them du lieu moi vao file hien tai")

    # Cho phép người dùng nhập số mẫu
    while True:
        try:
            target_samples = int(input("\nNhap so mau can thu thap cho moi tu the (mac dinh 2000): ") or "2000")
            if target_samples > 0:
                break
            print("So mau phai lon hon 0!")
        except ValueError:
            print("Vui long nhap mot so nguyen hop le!")

    # Initialize MediaPipe Pose and other variables
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    current_idx = 0
    current_label = current_labels[current_idx]
    
    recording = False
    frame_count = 0
    samples_per_label = 0
    
    print(f"\nBat dau thu thap du lieu {data_type}...")
    print("Dat camera o ben PHAI cua ban")
    print(f"So mau can thu thap cho moi {data_type}: {target_samples}")
    
    # Hướng dẫn thu thập dữ liệu
    pose_instructions = {
        "good_sitting_side": "Ngoi thang lung, dau thang",
        "bad_sitting_backward_side": "Nga nguoi ra sau, lung tua ghe",
        "bad_sitting_forward_side": "Cui nguoi ve phia truoc, lung cong",
        "too_lean_left_side": "Nghieng nguoi sang TRAI (nhin tu phia camera)",
        "too_lean_right_side": "Nghieng nguoi sang PHAI (nhin tu phia camera)",
        "leg_right_position": "Chan dat song song, ban chan sat san",
        "leg_wrong_position": "Chan vat cheo, gac len ghe hoac duoi thang"
    }
    
    # Chọn và khởi tạo camera
    cap = choose_camera()
    if cap is None:
        print("Khong the khoi tao camera. Thoat chuong trinh.")
        return
    
    # Xác định nếu là camera WiFi
    is_wifi_camera = False
    if isinstance(cap, cv2.VideoCapture) and cap.isOpened():
        properties = [cap.get(prop) for prop in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS]]
        # Camera WiFi thường có kích thước khác với camera mặc định
        if properties[0] != 1280 or properties[1] != 720:
            is_wifi_camera = True
            print("Da phat hien camera WiFi - Dieu chinh kich thuoc chu.")
    
    # Main loop
    print("\nSan sang thu thap du lieu!")
    print("Nhan 'r' de bat dau/dung ghi hinh, 'n' de chuyen nhan, 'q' de thoat")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get pose results and keypoints
        results = get_pose_results(frame)[0]
        keypoints = extract_keypoints(frame)
        
        # Tính góc nghiêng
        lean_angle = 0
        if results and results.pose_landmarks:
            lean_angle = calculate_lean_angle(results)
        
        # Draw landmarks on frame
        if results and results.pose_landmarks:
            frame = draw_landmarks(frame, results)
            
            if keypoints is not None and recording:
                if samples_per_label < target_samples:
                    # Save data based on mode
                    save_to_csv(keypoints, current_label, is_leg_data, is_neck_data)
                    save_raw(frame, current_label, frame_count)
                    frame_count += 1
                    samples_per_label += 1
                else:
                    recording = False
                    print(f"\nHoan thanh thu thap {target_samples} mau cho {current_label}")
                    current_idx = (current_idx + 1) % len(current_labels)
                    current_label = current_labels[current_idx]
                    samples_per_label = 0

        # Add recording status and current label to frame
        status = "DANG GHI HINH" if recording else "DUNG GHI HINH"
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        
        # Điều chỉnh cỡ chữ cho trạng thái dựa trên loại camera
        font_scale = 0.40 if not is_wifi_camera else 0.35
        thickness = 1
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
        
        # Hiển thị nhãn hiện tại và số mẫu
        cv2.putText(frame, f"{data_type.upper()}: {current_label}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"SO MAU: {samples_per_label}/{target_samples}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Hiển thị hướng dẫn
        instruction = pose_instructions.get(current_label, "")
        cv2.putText(frame, f"HUONG DAN: {instruction}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 255), thickness)

        # Vẽ thông tin điều khiển
        draw_controls_info(frame, recording, is_wifi_camera)

        # Vẽ trợ giúp và đồng hồ góc nghiêng
        frame = draw_lean_guide(frame, current_label, lean_angle, is_wifi_camera)

        # Show frame
        cv2.namedWindow("Thu thap du lieu", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Thu thap du lieu", 1280, 720)
        cv2.imshow("Thu thap du lieu", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            if recording:
                print(f"\nBat dau ghi hinh {current_label}")
            else:
                print(f"Dung ghi hinh {current_label}")
        elif key == ord("n"):
            recording = False
            current_idx = (current_idx + 1) % len(current_labels)
            current_label = current_labels[current_idx]
            samples_per_label = 0
            print(f"\nChuyen sang {data_type}: {current_label}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nHoan thanh thu thap du lieu!")
    print(f"Tong so frame da thu thap: {frame_count}")
    if mode == 1:
        print("Du lieu tu the da duoc luu vao data/processed/posture_dataset.csv")
    elif mode == 2:
        print("Du lieu vi tri chan da duoc luu vao data/processed/leg_dataset.csv")
    else:  # mode == 3
        print("Du lieu tu the co da duoc luu vao data/processed/neck_dataset.csv")

if __name__ == "__main__":
    main()