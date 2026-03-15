import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import time

# 1. Load Haar Cascade (cross-platform, no Windows path)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(cascade_path)

if face_classifier.empty():
    print(f"❌ Haar cascade failed to load from: {cascade_path}")
    exit()

# 2. Load Emotion Detection Model
model_path = "model.h5"
if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found in directory: {os.getcwd()}")
    exit() \
        

classifier = load_model(model_path)

# 3. Emotion Labels
emotion_labels = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

# 4. Start Camera (Windows-friendly)
# Try DirectShow backend (Windows default)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Fallback to default backend if that fails
if not cap.isOpened():
    print("⚠️ Could not open camera with DirectShow, trying default...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot access camera. Check Windows permissions.")
    print("➡️ Go to Windows Settings → Privacy & Security → Camera → Allow access.")
    exit()

print("✅ Camera started. Press 'q' to quit.")
time.sleep(1.0)  # Give macOS a second to initialize the camera

# 5. Main Loop
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("⚠️ Failed to read frame. Retrying...")
        time.sleep(0.2)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) == 0:
            continue

        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi, verbose=0)[0]
        label = emotion_labels[prediction.argmax()]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("😊 Emotion Detector (Press 'q' to quit)", frame)

    # Break if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 Exiting...")
        break


# 6. Cleanup

cap.release()
cv2.destroyAllWindows()
