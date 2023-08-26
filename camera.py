import cv2
import websockets
import asyncio
import numpy as np
import threading

async def send_frames(cap):
    uri = "ws://localhost:4100/"
    async with websockets.connect(uri) as websocket:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame")
                break

            # Kodieren Sie das Frame als JPEG
            _, buffer = cv2.imencode('.jpg', frame)

            # Senden Sie die JPEG-Daten über WebSocket
            await websocket.send(buffer.tobytes())

def init_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Fehler: Konnte Kamera {index} nicht öffnen.")
        return None
    return cap

def start_websocket_thread(cap):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_frames(cap))

cap1 = init_camera(0)
cap2 = init_camera(1)

if cap1 is None or cap2 is None:
    exit()

# Starte den WebSocket-Thread für die erste Kamera
websocket_thread = threading.Thread(target=start_websocket_thread, args=(cap1,))
websocket_thread.start()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Fehler beim Erfassen der Frames.")
        break

    cv2.imshow('Kamera 1', frame1)
    cv2.imshow('Kamera 2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
