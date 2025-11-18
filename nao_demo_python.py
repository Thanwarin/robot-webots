# emotion_controller.py
from controller import Robot, Camera, Motor
import numpy as np
import cv2
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

print("Python executable:", sys.executable)
print("TensorFlow version:", tf.__version__)
print("Numpy version:", np.__version__)

# --- PARAMETERS ---
TIME_STEP = 32  # Webots simulation timestep
MODEL_PATH = "emotion_detection_pretrain_RetinaFace_FERplus.h5"  # path ของ pre-trained emotion model
IMG_WIDTH, IMG_HEIGHT = 48, 48  # ขนาด input ของ model
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']  # emotion labels

# --- INITIALIZE ROBOT ---
robot = Robot()

# Camera
camera = robot.getDevice("CameraTop")  # ใช้ camera ของ NAO
camera.enable(TIME_STEP)

# Motors (หัวและแขน)
head_yaw = robot.getDevice("HeadYaw")
head_pitch = robot.getDevice("HeadPitch")
r_shoulder_pitch = robot.getDevice("RShoulderPitch")
r_shoulder_roll = robot.getDevice("RShoulderRoll")

# --- LOAD MODEL ---
model = load_model(MODEL_PATH)

# --- MEDIAPIPE FACE DETECTION ---
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- MAIN LOOP ---
while robot.step(TIME_STEP) != -1:
    # 1. Capture image
    image = camera.getImage()
    if image is None:
        continue

    # แปลงภาพจาก Webots BGRA -> BGR
    img_bgr = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    # --- FACE DETECTION ---
    rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_image)

    if results.detections:
        # มีใบหน้า
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            h, w, _ = img_bgr.shape
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))

            face_crop = img_bgr[y1:y2, x1:x2]

            # preprocess สำหรับ model
            face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT))
            face_input = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_input, axis=0)  # (1,48,48,3)

            # predict emotion
            preds = model.predict(face_input, verbose=0)
            emotion_index = np.argmax(preds)
            emotion = EMOTION_LABELS[emotion_index]
            print("Detected emotion:", emotion)

            # วาด bounding box และ label
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_bgr, emotion, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # --- CONTROL ROBOT BASED ON EMOTION ---
            if emotion == 'Happy':
                r_shoulder_pitch.setPosition(1.0)
                r_shoulder_roll.setPosition(0.5)
            elif emotion == 'Sad':
                head_pitch.setPosition(0.5)
            elif emotion == 'Angry':
                head_yaw.setPosition(0.5)
            else:
                # Neutral / Surprise / etc -> reset pose
                r_shoulder_pitch.setPosition(0.0)
                r_shoulder_roll.setPosition(0.0)
                head_pitch.setPosition(0.0)
                head_yaw.setPosition(0.0)
    else:
        # ไม่พบหน้า
        print("No face detected")
        # reset robot pose
        r_shoulder_pitch.setPosition(0.0)
        r_shoulder_roll.setPosition(0.0)
        head_pitch.setPosition(0.0)
        head_yaw.setPosition(0.0)

    # --- SHOW IMAGE ---
    cv2.imshow("NAO Camera View", img_bgr)
    cv2.waitKey(1)  # ต้องเรียกทุกครั้งเพื่อ refresh window
