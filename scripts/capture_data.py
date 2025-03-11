import cv2
import os
import numpy as np
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.helpers import save_to_csv, save_raw
from utils.visualization import draw_landmarks
import csv
import time
import shutil

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
    """Chỉ lấy các keypoint phần thân trên (không bao gồm chân)"""
    # Loại bỏ các keypoint của chân (từ 23-32)
    upper_indices = list(range(0, 23))  # 0-22: từ đầu đến hông
    upper_keypoints = []
    
    for idx in upper_indices:
        start_idx = idx * 3
        upper_keypoints.extend(keypoints[start_idx:start_idx + 3])
    
    return np.array(upper_keypoints)

def save_to_csv(keypoints, label, is_leg_data=False, csv_file=None):
    """Save keypoints to CSV file with proper formatting"""
    if csv_file is None:
        csv_file = "data/processed/leg_dataset.csv" if is_leg_data else "data/processed/posture_dataset.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Process keypoints based on data type
    if is_leg_data:
        keypoints = extract_leg_keypoints(keypoints)
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
            else:
                header = [f'kp_{i}_{ax}' for i in range(23) for ax in ['x', 'y', 'z']]
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

def draw_controls_info(frame, recording):
    """Vẽ thông tin các phím điều khiển"""
    # Vẽ background cho controls
    controls_height = 120
    controls_width = 300
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (frame.shape[1] - controls_width - 10, 10),
                 (frame.shape[1] - 10, controls_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Hiển thị các phím điều khiển
    x_pos = frame.shape[1] - controls_width
    y_pos = 40
    controls = [
        ("PHIM R", "BAT/TAT GHI HINH", (0, 255, 0) if recording else (0, 0, 255)),
        ("PHIM N", "DOI TU THE", (255, 255, 255)),
        ("PHIM Q", "THOAT", (255, 255, 255))
    ]
    
    for key, action, color in controls:
        cv2.putText(frame, f"{key}: {action}", 
                    (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        y_pos += 30

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
    
    # Mode selection
    print("\nChọn chế độ thu thập dữ liệu:")
    print("1: Thu thập dữ liệu tư thế")
    print("2: Thu thập dữ liệu vị trí chân")
    while True:
        try:
            mode = int(input("Nhập lựa chọn (1 hoặc 2): "))
            if mode in [1, 2]:
                break
            print("Vui lòng nhập 1 hoặc 2!")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ!")

    # Set up based on mode
    if mode == 1:
        current_labels = postures
        is_leg_data = False
        data_type = "tư thế"
        dataset_file = "data/processed/posture_dataset.csv"
    else:
        current_labels = leg_positions
        is_leg_data = True
        data_type = "vị trí chân"
        dataset_file = "data/processed/leg_dataset.csv"
    
    # Handle existing dataset file
    if os.path.exists(dataset_file):
        print(f"\nĐã tìm thấy file dữ liệu {data_type} cũ.")
        print("Bạn muốn xử lý file cũ như thế nào?")
        print("1: Ghi đè (xóa dữ liệu cũ)")
        print("2: Tạo backup và bắt đầu file mới")
        print("3: Giữ nguyên và thêm dữ liệu vào file cũ")
        
        while True:
            try:
                file_choice = int(input("Nhập lựa chọn của bạn (1-3): "))
                if file_choice in [1, 2, 3]:
                    break
                print("Vui lòng nhập 1, 2 hoặc 3!")
            except ValueError:
                print("Vui lòng nhập một số hợp lệ!")
        
        if file_choice == 1:
            os.remove(dataset_file)
            print(f"Đã xóa file dữ liệu {data_type} cũ")
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
            print(f"Đã sao lưu file cũ vào: {backup_file}")
            print(f"Đã tạo file mới để ghi dữ liệu")
        else:
            print(f"Sẽ thêm dữ liệu mới vào file hiện tại")

    # Cho phép người dùng nhập số mẫu
    while True:
        try:
            target_samples = int(input("\nNhap so mau can thu thap cho moi tu the (mac dinh 2000): ") or "2000")
            if target_samples > 0:
                break
            print("So mau phai lon hon 0!")
        except ValueError:
            print("Vui long nhap mot so nguyen hop le!")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_idx = 0
    current_label = current_labels[current_idx]
    
    recording = False
    frame_count = 0
    samples_per_label = 0
    
    print(f"\nBắt đầu thu thập dữ liệu {data_type}...")
    print("Đặt camera ở bên PHẢI của bạn")
    print(f"Số mẫu cần thu thập cho mỗi {data_type}: {target_samples}")
    
    # Instructions based on mode
    if mode == 1:
        pose_instructions = {
            "good_sitting_side": "Ngồi thẳng lưng, đầu thẳng",
            "bad_sitting_backward_side": "Ngả người ra sau, lưng tựa ghế",
            "bad_sitting_forward_side": "Cúi người về phía trước, lưng cong",
            "too_lean_left_side": "Nghiêng người sang trái",
            "too_lean_right_side": "Nghiêng người sang phải"
        }
    else:
        pose_instructions = {
            "leg_right_position": "Hai chân đặt song song, bàn chân đặt sát sàn",
            "leg_wrong_position": "Chân đặt sai tư thế (vắt chéo, gác lên ghế, duỗi thẳng...)"
        }

    print(f"\nCác {data_type} cần thu thập:")
    for label, instruction in pose_instructions.items():
        print(f"- {label}: {instruction}")
    
    print("\nLưu ý khi thu thập dữ liệu:")
    print("- Di chuyển nhẹ để có sự đa dạng trong dữ liệu")
    print("- Xoay người nhẹ (±15 độ) để tăng tính đa dạng")
    print("- Giữ nguyên tư thế cơ bản")
    print("- Đảm bảo ánh sáng tốt")
    print("- Giữ nguyên vị trí camera ở bên phải")
    print("- Đảm bảo nhìn thấy toàn bộ cơ thể trong khung hình")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get pose results and keypoints
        results = get_pose_results(frame)[0]
        keypoints = extract_keypoints(frame)
        
        # Draw landmarks on frame
        if results and results.pose_landmarks:
            frame = draw_landmarks(frame, results)
            
            if keypoints is not None and recording:
                if samples_per_label < target_samples:
                    # Save data based on mode
                    save_to_csv(keypoints, current_label, is_leg_data)
                    save_raw(frame, current_label, frame_count)
                    frame_count += 1
                    samples_per_label += 1
                else:
                    recording = False
                    print(f"\nHoàn thành thu thập {target_samples} mẫu cho {current_label}")
                    current_idx = (current_idx + 1) % len(current_labels)
                    current_label = current_labels[current_idx]
                    samples_per_label = 0

        # Add recording status and current label to frame
        status = "ĐANG GHI HÌNH" if recording else "DỪNG GHI HÌNH"
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Hiển thị nhãn hiện tại và số mẫu
        cv2.putText(frame, f"{data_type.upper()}: {current_label}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"SỐ MẪU: {samples_per_label}/{target_samples}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Hiển thị hướng dẫn
        instruction = pose_instructions.get(current_label, "")
        cv2.putText(frame, f"HƯỚNG DẪN: {instruction}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Vẽ thông tin điều khiển
        draw_controls_info(frame, recording)

        # Show frame
        cv2.namedWindow("Thu thập dữ liệu", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Thu thập dữ liệu", 1280, 720)
        cv2.imshow("Thu thập dữ liệu", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            if recording:
                print(f"\nBắt đầu ghi hình {current_label}")
            else:
                print(f"Dừng ghi hình {current_label}")
        elif key == ord("n"):
            recording = False
            current_idx = (current_idx + 1) % len(current_labels)
            current_label = current_labels[current_idx]
            samples_per_label = 0
            print(f"\nChuyển sang {data_type}: {current_label}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nHoàn thành thu thập dữ liệu!")
    print(f"Tổng số frame đã thu thập: {frame_count}")
    if mode == 1:
        print("Dữ liệu tư thế đã được lưu vào data/processed/posture_dataset.csv")
    else:
        print("Dữ liệu vị trí chân đã được lưu vào data/processed/leg_dataset.csv")

if __name__ == "__main__":
    main()