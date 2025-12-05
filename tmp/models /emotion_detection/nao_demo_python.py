# ==============================
# emotion_controller_webcam.py
# ==============================
from controller import Robot, Motor
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# ==============================
# PARAMETERS
# ==============================
TIME_STEP = 32
EMOTION_MODEL_PATH = "emotion_model_ver6.h5"  # โหลดโมเดล emotion classifier
IMG_WIDTH, IMG_HEIGHT = 96, 96
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ==============================
# INITIALIZE ROBOT
# ==============================
robot = Robot()

# Motors
head_yaw = robot.getDevice("HeadYaw")
head_pitch = robot.getDevice("HeadPitch")
r_shoulder_pitch = robot.getDevice("RShoulderPitch")
r_shoulder_roll = robot.getDevice("RShoulderRoll")

# ==============================
# LOAD EMOTION MODEL
# ==============================
model = load_model(EMOTION_MODEL_PATH)

# ==============================
# MEDIAPIPE FACE DETECTION
# ==============================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ==============================
# FUNCTION TO PREDICT EMOTION
# ==============================
def predict_emotion(face_img):
    face_resized = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
    face_input = face_resized.astype("float32") / 255.0
    face_input = np.expand_dims(face_input, axis=0)  # (1,96,96,3)
    preds = model.predict(face_input, verbose=0)
    emotion_index = np.argmax(preds)
    return EMOTION_LABELS[emotion_index]

# ==============================
# INITIALIZE WEBCAM
# ==============================
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam ready!")

# ==============================
# MAIN LOOP
# ==============================
while robot.step(TIME_STEP) != -1:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))

            face_crop = frame[y1:y2, x1:x2]
            emotion = predict_emotion(face_crop)
            print("Detected emotion:", emotion)

            # Draw box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, emotion, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # ==============================
            # CONTROL NAO MOTORS BASED ON EMOTION
            # ==============================
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
        print("No face detected")
        # reset pose
        r_shoulder_pitch.setPosition(0.0)
        r_shoulder_roll.setPosition(0.0)
        head_pitch.setPosition(0.0)
        head_yaw.setPosition(0.0)

    # Display webcam frame
    cv2.imshow("Webcam NAO View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
