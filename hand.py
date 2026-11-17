#imports

import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import urllib.request

#Get googles model for hand marking
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

#Make it mark the hands with frames each given their own timestamp, like an animation and stich together at the end

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5, #Coin flip test
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

#0.5 to make sure it is at least more probable than simply randomly guessing that there is a hand in the frame, and that it is actually a hand and not some other object

#BONE TIME
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

#TIPS
TIPS = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

#open up the webcam and capture some frames
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

start = time.time()
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1) #Need to mirror it cuz camera is weirddd
    h, w = frame.shape[:2]

    # Convert the frame to RGB format
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int((time.time() - start) * 1000) # Get the current timestamp in milliseconds
    result = landmarker.detect_for_video(mp_image, timestamp_ms)


    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        side = result.handedness[0][0].category_name

        pts = [(int(p.x * w), int(p.y * h)) for p in hand]

        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (x, y), 5, (0, 200, 255), -1)
            cv2.putText(frame, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        for i, name in TIPS.items():
            cv2.putText(frame, name, (pts[i][0] - 15, pts[i][1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.putText(frame, f"{side} hand", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #Count the FPS
    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Landmarker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
