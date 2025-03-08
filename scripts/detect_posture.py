import cv2
import numpy as np
import tensorflow as tf
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.visualization import draw_landmarks
import pickle
import time
import os

def get_all_predictions(prediction, label_encoder):
    """Get all postures predictions and their confidence levels"""
    # Debug information
    print("\nDebug - Raw predictions:", prediction[0])
    print("Debug - Available classes:", label_encoder.classes_)
    
    # Lấy tư thế có confidence cao nhất
    max_idx = np.argmax(prediction[0])
    max_conf = prediction[0][max_idx]
    posture = label_encoder.classes_[max_idx]
    
    print(f"Debug - Selected index: {max_idx}")
    print(f"Debug - Selected posture: {posture}")
    print(f"Debug - Confidence: {max_conf:.2%}")
    
    # Convert posture to string to avoid display issues
    return [(str(posture), max_conf)]

def get_posture_display_name(posture):
    """Chuyển đổi tên posture thành text hiển thị chi tiết"""
    # Handle numeric postures
    if isinstance(posture, (int, np.integer)) or posture.isdigit():
        posture_map = {
            0: "DANG NGOI THANG LUNG",
            1: "DANG CUI NGUOI VE PHIA TRUOC",
            2: "DANG NGA NGUOI RA SAU",
            3: "DANG NGHIENG NGUOI SANG TRAI",
            4: "DANG NGHIENG NGUOI SANG PHAI",
            5: "DANG NGOI VAT CHEO CHAN"
        }
        return posture_map.get(int(posture), f"TU THE {posture}")
    
    # Original mapping for string postures
    posture_names = {
        "good_sitting_side": "DANG NGOI THANG LUNG",
        "bad_sitting_forward_side": "DANG CUI NGUOI VE PHIA TRUOC",
        "bad_sitting_backward_side": "DANG NGA NGUOI RA SAU",
        "too_lean_left_side": "DANG NGHIENG NGUOI SANG TRAI",
        "too_lean_right_side": "DANG NGHIENG NGUOI SANG PHAI",
        "legs_crossed": "DANG NGOI VAT CHEO CHAN"
    }
    return posture_names.get(posture, str(posture))

def get_posture_improvement(posture):
    """Trả về hướng dẫn cải thiện tư thế"""
    # Handle numeric postures
    if isinstance(posture, (int, np.integer)) or str(posture).isdigit():
        posture_map = {
            0: "good_sitting_side",
            1: "bad_sitting_forward_side",
            2: "bad_sitting_backward_side",
            3: "too_lean_left_side",
            4: "too_lean_right_side",
            5: "legs_crossed"
        }
        posture = posture_map.get(int(posture), str(posture))
    
    improvements = {
        "good_sitting_side": [
            "- Tiep tuc giu nguyen tu the nay",
            "- Dam bao lung thang, dau thang",
            "- Hai chan dat san"
        ],
        0: [  # Numeric mapping
            "- Tiep tuc giu nguyen tu the nay",
            "- Dam bao lung thang, dau thang",
            "- Hai chan dat san"
        ],
        "bad_sitting_forward_side": [
            "- Tu tu ngoi thang lung len",
            "- Keo vai ra sau",
            "- Nang dau len",
            "- Tranh cong lung khi lam viec"
        ],
        1: [  # Numeric mapping
            "- Tu tu ngoi thang lung len",
            "- Keo vai ra sau",
            "- Nang dau len",
            "- Tranh cong lung khi lam viec"
        ],
        "bad_sitting_backward_side": [
            "- Dieu chinh lung ve phia truoc",
            "- Giu thang lung",
            "- Tranh nga nguoi ra sau qua lau"
        ],
        2: [  # Numeric mapping
            "- Dieu chinh lung ve phia truoc",
            "- Giu thang lung",
            "- Tranh nga nguoi ra sau qua lau"
        ]
    }
    return improvements.get(posture, ["- Hay dieu chinh tu the cho phu hop"])

def get_posture_recommendation(posture):
    """Trả về khuyến nghị cho từng tư thế"""
    recommendations = {
        "good_sitting_side": "Giu nguyen tu the nay, rat tot cho cot song",
        "bad_sitting_forward_side": "Hay ngoi thang lung len, tranh cong lung",
        "bad_sitting_backward_side": "Dieu chinh lung thang dung, tranh nga nguoi",
        "too_lean_left_side": "Dieu chinh lai tu the, tranh nghieng sang trai",
        "too_lean_right_side": "Dieu chinh lai tu the, tranh nghieng sang phai",
        "legs_crossed": "Nen dat ca 2 chan xuong san, tranh vat cheo"
    }
    return recommendations.get(posture, "")

def get_posture_severity(posture):
    """Xác định mức độ nghiêm trọng của tư thế"""
    severity_levels = {
        "good_sitting_side": "TOT",
        "bad_sitting_forward_side": "TRUNG BINH",
        "bad_sitting_backward_side": "TRUNG BINH",
        "too_lean_left_side": "NGUY HIEM",
        "too_lean_right_side": "NGUY HIEM",
        "legs_crossed": "TRUNG BINH"
    }
    return severity_levels.get(posture, "KHONG XAC DINH")

def get_posture_status(confidence):
    """Xác định trạng thái tư thế dựa trên độ tin cậy"""
    if confidence >= 0.8:
        return "TOT", (0, 255, 0)  # Xanh lá
    elif confidence >= 0.5:
        return "CANH BAO", (0, 255, 255)  # Vàng
    else:
        return "KHONG TOT", (0, 0, 255)  # Đỏ

def draw_prediction_info(frame, postures):
    """Vẽ thông tin dự đoán và hướng dẫn cải thiện"""
    # Vẽ background cho thông tin
    info_height = 300
    info_width = 600
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (info_width, info_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_offset = 40
    
    # Hiển thị tư thế hiện tại và độ chính xác
    posture, conf = postures[0]
    display_name = get_posture_display_name(posture)
    
    # Debug information
    print(f"\nDebug - Detected posture: {posture}")
    print(f"Debug - Display name: {display_name}")
    print(f"Debug - Confidence: {conf:.2%}")
    
    # Hiển thị tư thế với màu dựa trên confidence
    color = (0, 255, 0) if conf >= 0.8 else (0, 255, 255) if conf >= 0.5 else (0, 0, 255)
    cv2.putText(frame, f"TU THE HIEN TAI:", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2)
    y_offset += 30
    
    cv2.putText(frame, str(display_name), 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, color, 2)
    y_offset += 30
    
    cv2.putText(frame, f"DO CHINH XAC: {conf:.1%}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    y_offset += 50
    
    # Hiển thị hướng dẫn cải thiện
    cv2.putText(frame, "HUONG DAN CAI THIEN:", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    y_offset += 30
    
    improvements = get_posture_improvement(posture)
    for improvement in improvements:
        cv2.putText(frame, str(improvement), 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        y_offset += 30

def main():
    # Check required files
    required_files = [
        ("models/pose_classifier.h5", "Model file"),
        ("models/label_encoder.pkl", "Label encoder file"),
        ("models/normalization_mean.npy", "Normalization mean file"),
        ("models/normalization_std.npy", "Normalization std file")
    ]
    
    for file_path, file_desc in required_files:
        if not os.path.exists(file_path):
            print(f"Error: {file_desc} not found at {file_path}")
            print("Please run training first using: python scripts/train_model.py")
            return

    # Load model and label encoder
    print("\n=== KHOI TAO HE THONG ===")
    try:
        model = tf.keras.models.load_model("models/pose_classifier.h5")
        print("- Da tai model thanh cong")
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        print("\nCau truc model:")
        print("\n".join(model_summary))
        
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print("\n- Da tai label encoder thanh cong")
        print("\nCac tu the co the nhan dang:")
        for idx, class_name in enumerate(label_encoder.classes_):
            print(f"- {idx}: {get_posture_display_name(class_name)}")
        
        # Load normalization parameters
        X_train_mean = np.load("models/normalization_mean.npy")
        X_train_std = np.load("models/normalization_std.npy")
        print("\n- Da tai thong so chuan hoa thanh cong")
        print(f"- Shape mean: {X_train_mean.shape}")
        print(f"- Shape std: {X_train_std.shape}")
        print(f"- Mean range: [{np.min(X_train_mean):.2f}, {np.max(X_train_mean):.2f}]")
        print(f"- Std range: [{np.min(X_train_std):.2f}, {np.max(X_train_std):.2f}]")
        
    except Exception as e:
        print(f"\nLoi khi tai model: {str(e)}")
        print("Hay chay lai training")
        return

    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Loi: Khong the mo webcam")
            return
        print("\n- Da khoi tao webcam thanh cong")
    except Exception as e:
        print(f"Loi khoi tao webcam: {str(e)}")
        return

    print("\n=== BAT DAU NHAN DANG ===")
    print("Nhan 'q' de thoat")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Loi: Khong the doc frame tu webcam")
            break

        frame_count += 1
        debug_frame = frame_count % 30 == 0  # Debug mỗi 30 frames
        
        if debug_frame:
            print(f"\nFrame #{frame_count}")
            print(f"Frame shape: {frame.shape}")

        # Get keypoints
        keypoints = extract_keypoints(frame)
        
        # Draw landmarks
        results = get_pose_results(frame)[0]
        if results:
            frame = draw_landmarks(frame, results)
            if debug_frame:
                print("- Da phat hien pose landmarks")
                if keypoints is not None:
                    print(f"- So keypoints phat hien: {len(keypoints)}")
                    print(f"- Keypoints range: [{np.min(keypoints):.2f}, {np.max(keypoints):.2f}]")

        if keypoints is not None:
            try:
                # Preprocess keypoints
                keypoints_normalized = (keypoints - X_train_mean) / X_train_std
                keypoints_reshaped = keypoints_normalized.reshape(1, -1)
                
                if debug_frame:
                    print("\nPreprocessing details:")
                    print(f"- Raw keypoints range: [{np.min(keypoints):.2f}, {np.max(keypoints):.2f}]")
                    print(f"- Normalized range: [{np.min(keypoints_normalized):.2f}, {np.max(keypoints_normalized):.2f}]")
                    print(f"- Expected input shape: {model.input_shape}")
                    print(f"- Actual input shape: {keypoints_reshaped.shape}")
                
                # Model predictions
                predictions = model.predict(keypoints_reshaped, verbose=0)
                
                if debug_frame:
                    print("\nPrediction details:")
                    print("- Raw predictions per class:")
                    for idx, (class_name, pred) in enumerate(zip(label_encoder.classes_, predictions[0])):
                        print(f"  {idx}: {get_posture_display_name(class_name)}: {pred:.4f}")
                    print(f"- Sum predictions: {np.sum(predictions[0]):.4f}")
                    print(f"- Max confidence: {np.max(predictions[0]):.2%}")
                
                postures = get_all_predictions(predictions, label_encoder)
                
                # Draw prediction info
                draw_prediction_info(frame, postures)

            except Exception as e:
                print(f"\nLoi khi du doan: {str(e)}")
                print("Stack trace:")
                import traceback
                traceback.print_exc()
                cv2.putText(frame, "Loi: Khong the nhan dang tu the", 
                           (20, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if debug_frame:
                print("- Khong phat hien duoc keypoints")
                print("- Hay dam bao:")
                print("  1. Ban dung trong khung hinh")
                print("  2. Anh sang du sang")
                print("  3. Khong co vat can chan tam nhin")

        # Show frame
        cv2.namedWindow("Nhan Dang Tu The", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Nhan Dang Tu The", 1280, 720)
        cv2.imshow("Nhan Dang Tu The", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== KET THUC NHAN DANG ===")

if __name__ == "__main__":
    main()