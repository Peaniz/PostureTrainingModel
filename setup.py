import os

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        "data/raw",
        "data/processed",
        "data/splits",
        "data/training_plots",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    setup_directories() 