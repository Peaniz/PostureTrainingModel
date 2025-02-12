import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Body Posture Detection")
    parser.add_argument("--mode", type=str, required=True, help="Mode: capture, train, or detect")
    args = parser.parse_args()

    if args.mode == "capture":
        from scripts.capture_data import main as capture_main
        capture_main()
    elif args.mode == "train":
        from scripts.train_model import main as train_main
        train_main()
    elif args.mode == "detect":
        from scripts.detect_posture import main as detect_main
        detect_main()
    else:
        print("Invalid mode. Use 'capture', 'train', or 'detect'.")