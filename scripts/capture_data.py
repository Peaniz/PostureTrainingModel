import cv2
import os
import numpy as np
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.helpers import save_to_csv, save_raw
from utils.visualization import draw_landmarks
import csv
import time

def save_to_csv(keypoints, label, csv_file="data/processed/dataset.csv"):
    """Save keypoints to CSV file with proper formatting"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
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
            header = [f'kp_{i}_{ax}' for i in range(33) for ax in ['x', 'y', 'z']]
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
    
    # Delete existing dataset file to start fresh
    dataset_file = "data/processed/dataset.csv"
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
        print("Removed existing dataset file")

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
    
    # Define posture classes
    postures = [
        "good_sitting_side",          # Ngồi thẳng lưng
        "bad_sitting_forward_side",   # Cúi người về phía trước
        "bad_sitting_backward_side",  # Ngả người ra sau
        "too_lean_left_side",        # Nghiêng người sang trái (xa camera)
        "too_lean_right_side",       # Nghiêng người sang phải (gần camera)
        "legs_crossed"               # Ngồi vắt chéo chân
    ]
    
    current_posture_idx = 0
    pose_label = postures[current_posture_idx]
    
    recording = False
    frame_count = 0
    samples_per_posture = 0
    
    print("\nBat dau thu thap du lieu...")
    print("Dat camera o ben PHAI cua ban")
    print(f"So mau can thu thap cho moi tu the: {target_samples}")
    print("\nHuong dan cho tung tu the:")
    pose_instructions = {
        "good_sitting_side": "Ngoi thang lung, dau thang, chan dat san",
        "bad_sitting_forward_side": "Cui nguoi ve phia truoc, lung cong",
        "bad_sitting_backward_side": "Nga nguoi ra sau",
        "too_lean_left_side": "Nghieng nguoi sang trai (ra xa camera)",
        "too_lean_right_side": "Nghieng nguoi sang phai (ve phia camera)",
        "legs_crossed": "Ngoi vat cheo chan"
    }
    
    print("\nCac tu the can thu thap:")
    for pose, instruction in pose_instructions.items():
        print(f"- {pose}: {instruction}")
    
    print("\nLuu y khi thu thap du lieu:")
    print("- Di chuyen nhe de co su da dang trong du lieu")
    print("- Xoay nguoi nhe (±15 do) de tang tinh da dang")
    print("- Giu nguyen tu the co ban")
    print("- Dam bao anh sang tot")
    print("- Giu nguyen vi tri camera o ben phai")
    print("- Dam bao nhin thay toan bo co the trong khung hinh")
    print("\nCac phim chuc nang:")
    print("- 'r': Bat/tat ghi hinh")
    print("- 'n': Chuyen sang tu the tiep theo")
    print("- 'q': Thoat chuong trinh")

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
                if samples_per_posture < target_samples:
                    # Save data
                    save_to_csv(keypoints, pose_label)
                    save_raw(frame, pose_label, frame_count)
                    frame_count += 1
                    samples_per_posture += 1
                else:
                    recording = False
                    print(f"\nHoan thanh thu thap {target_samples} mau cho {pose_label}")
                    current_posture_idx = (current_posture_idx + 1) % len(postures)
                    pose_label = postures[current_posture_idx]
                    samples_per_posture = 0

        # Add recording status and current posture to frame
        status = "DANG GHI HINH" if recording else "DUNG GHI HINH"
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Hiển thị tư thế hiện tại và số mẫu
        cv2.putText(frame, f"TU THE: {pose_label}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"SO MAU: {samples_per_posture}/{target_samples}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Hiển thị hướng dẫn tư thế
        instruction = pose_instructions.get(pose_label, "")
        cv2.putText(frame, f"HUONG DAN: {instruction}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Vẽ thông tin điều khiển - truyền thêm biến recording
        draw_controls_info(frame, recording)

        # Show frame
        cv2.namedWindow("Thu thap du lieu tu the", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Thu thap du lieu tu the", 1280, 720)
        cv2.imshow("Thu thap du lieu tu the", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            if recording:
                print(f"\nBat dau ghi hinh {pose_label}")
            else:
                print(f"Dung ghi hinh {pose_label}")
        elif key == ord("n"):
            recording = False
            current_posture_idx = (current_posture_idx + 1) % len(postures)
            pose_label = postures[current_posture_idx]
            samples_per_posture = 0
            print(f"\nChuyen sang tu the {pose_label}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nHoan thanh thu thap du lieu!")
    print(f"Tong so frame da thu thap: {frame_count}")
    print("Du lieu da duoc luu vao data/processed/dataset.csv")

if __name__ == "__main__":
    main()