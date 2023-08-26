import cv2
import websockets
import asyncio
import numpy as np
import threading
import base64


async def send_frames():
    while True:
        uri = "wss://bernhackt23-backend.web01.dalcloud.net/"
        try:
            async with websockets.connect(uri) as websocket:
                cap = init_camera(0)
                if cap is None:
                    print("Failed to initialize camera. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to get frame")
                        break

                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_frame = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send(base64_frame)
                    await asyncio.sleep(0.1)

            cap.release()

        except Exception as e:
            print(f"WebSocket Error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


def init_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Couldn't open camera {index}.")
        return None
    return cap


def start_websocket_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_frames())


def main():
    websocket_thread = threading.Thread(target=start_websocket_thread)
    websocket_thread.start()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
