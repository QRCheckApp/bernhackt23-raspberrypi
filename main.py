import asyncio
import subprocess
import atexit
from sound_detection import sound_main

camera_process = None


def activate_camera():
    global camera_process
    print("Activating camera...")
    camera_process = subprocess.Popen(['python3', 'camera.py'])


def deactivate_camera():
    global camera_process
    if camera_process:
        print("Deactivating camera...")
        camera_process.terminate()


def cleanup():
    print("Cleaning up...")
    deactivate_camera()


atexit.register(cleanup)


def combined_main():
    sound_main(activate_camera)


if __name__ == "__main__":
    combined_main()
