import os
import subprocess
import sys

def install_requirements():
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found. Please make sure it exists in the current directory.")
        sys.exit(1)
    
    print("Installing dependencies from requirements.txt...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
