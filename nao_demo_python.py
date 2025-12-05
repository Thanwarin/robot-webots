"""
NAO Emotion Controller with Learned Policy + Happy & Big Dance
- Python 3 + Webots API
- Uses Mediapipe Face Detection + Keras Emotion Model + Q-Learning Policy
- Smooth motor interpolation to prevent falling
"""

import os
import cv2
import numpy as np
import logging
from controller import Robot
from tensorflow.keras.models import load_model
import mediapipe as mp


# ------------------------------
# CONFIG
# ------------------------------
TIME_STEP = 40  # ms per simulation step
Q_TABLE_PATH = "nao_emotion_qtable_tuned.npy"
EMOTION_MODEL_PATH = "emotion_model_ver9.h5"

IMG_WIDTH, IMG_HEIGHT = 96, 96
EMOTION_LABELS_7 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
CONFIDENCE_THRESHOLD = 0.6

# Q-Learning states & actions
Q_STATE_NAMES = ['Neutral', 'Happy', 'Angry', 'Sad', 'Surprise']
Q_ACTION_NAMES = ['Patrol (Walk)', 'Wave', 'Stomp', 'Slouch', 'Big Dance', 'Happy Dance']  # Added Happy Dance

FRAME_SKIP = 3
WEBCAM_W, WEBCAM_H = 320, 240

logging.basicConfig(filename="nao_emotion_log.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------
# INITIALIZE ROBOT + MOTORS
# ------------------------------
robot = Robot()

# Motors (Webots NAO model)
motor_names = [
    "HeadYaw","HeadPitch",
    "LShoulderPitch","LShoulderRoll","RShoulderPitch","RShoulderRoll",
    "LHipPitch","RHipPitch","LKneePitch","RKneePitch",
    "LHipRoll","RHipRoll","LAnklePitch","RAnklePitch","LAnkleRoll","RAnkleRoll"
]
motors = {name: robot.getDevice(name) for name in motor_names}

# ------------------------------
# LOAD Q-TABLE
# ------------------------------
if os.path.exists(Q_TABLE_PATH):
    Q_TABLE = np.load(Q_TABLE_PATH)
    if Q_TABLE.shape != (len(Q_STATE_NAMES), len(Q_ACTION_NAMES)):
        logging.warning("Q-Table shape mismatch. Using zeros.")
        Q_TABLE = np.zeros((len(Q_STATE_NAMES), len(Q_ACTION_NAMES)))
else:
    logging.warning("Q-Table not found. Using zeros.")
    Q_TABLE = np.zeros((len(Q_STATE_NAMES), len(Q_ACTION_NAMES)))

# ------------------------------
# LOAD EMOTION MODEL
# ------------------------------
try:
    emotion_model = load_model(EMOTION_MODEL_PATH)
    logging.info("Loaded Keras model.")
except Exception as e:
    logging.exception("Failed to load Keras model: %s", e)
    raise SystemExit("Cannot load Keras model.")

# ------------------------------
# PATROL WALK SETUP
# ------------------------------
PATROL_SEQUENCE = [
    {"LHipPitch": 0.08, "RHipPitch": -0.08, "LKneePitch": 0.0, "RKneePitch": 0.0,
     "LAnklePitch": -0.04, "RAnklePitch": 0.04, "LHipRoll": 0.03, "RHipRoll": -0.03},
    {"LHipPitch": -0.08, "RHipPitch": 0.08, "LKneePitch": 0.0, "RKneePitch": 0.0,
     "LAnklePitch": 0.04, "RAnklePitch": -0.04, "LHipRoll": -0.03, "RHipRoll": 0.03}
]
patrol_step_index = 0
patrol_target_time = 0.0
PATROL_STEP_DURATION = 0.3  # slightly faster

def set_motor_targets(targets):
    """Set motors to target positions directly"""
    for name, val in targets.items():
        if name in motors:
            motors[name].setPosition(val)

def do_patrol_step():
    """Perform a patrol walk step with smooth interpolation"""
    global patrol_step_index, patrol_target_time
    if robot.getTime() >= patrol_target_time:
        current_targets = PATROL_SEQUENCE[patrol_step_index]
        for step in range(5):
            for name, val in current_targets.items():
                if name in motors:
                    motors[name].setPosition(val * (step + 1) / 5.0)
            robot.step(TIME_STEP)
        patrol_step_index = (patrol_step_index + 1) % len(PATROL_SEQUENCE)
        patrol_target_time = robot.getTime() + PATROL_STEP_DURATION

# ------------------------------
# UTILITIES
# ------------------------------
def map_keras_to_q_state(label):
    """Map Keras emotion label to Q-learning state index"""
    if label in ['Neutral', 'Disgust', 'Fear']:
        return 0
    elif label == 'Happy':
        return 1
    elif label == 'Angry':
        return 2
    elif label == 'Sad':
        return 3
    elif label == 'Surprise':
        return 4
    return 0

def decide_optimal_action(q_state):
    """Select optimal action based on Q-table"""
    q_values = Q_TABLE[q_state, :]
    action_idx = int(np.argmax(q_values))
    logging.info("Decision: Q-State=%s -> Action=%s", Q_STATE_NAMES[q_state], Q_ACTION_NAMES[action_idx])
    return action_idx

def smooth_motor_move(targets, steps=5):
    """Smoothly move motors to targets in 'steps' interpolation"""
    for i in range(1, steps + 1):
        alpha = i / float(steps)
        for name, val in targets.items():
            if name in motors:
                motors[name].setPosition(val * alpha)
        robot.step(TIME_STEP)

def reset_pose(steps=5):
    """Reset all motors to neutral pose"""
    targets = {name: 0.0 for name in motors.keys()}
    smooth_motor_move(targets, steps=steps)

def execute_nao_action(action_index):
    """Execute NAO action based on index"""
    reset_pose(steps=4)

    if action_index == 1:  # Wave
        targets = {"RShoulderPitch": 0.9, "RShoulderRoll": 0.4}
        smooth_motor_move(targets, steps=4)

    elif action_index == 2:  # Stomp (Angry)
        targets = {"HeadYaw": 0.25, "LHipPitch": 0.09, "RHipPitch": 0.09}
        smooth_motor_move(targets, steps=4)
        targets = {"LHipPitch": -0.09, "RHipPitch": -0.09}
        smooth_motor_move(targets, steps=4)

    elif action_index == 3:  # Slouch (Sad)
        targets = {"HeadPitch": 0.28}
        smooth_motor_move(targets, steps=4)

    elif action_index == 4:  # Big Dance (Surprise)
        # Big Dance: alternate arms, hips, head for 6 steps
        dance_targets = {}
        for i in range(6):
            dance_targets["LShoulderPitch"] = 0.4 * (-1)**i
            dance_targets["RShoulderPitch"] = 0.4 * (-1)**i
            dance_targets["LHipPitch"] = 0.15 * (-1)**i
            dance_targets["RHipPitch"] = 0.15 * (-1)**i
            dance_targets["HeadYaw"] = 0.25 * (-1)**i
            dance_targets["HeadPitch"] = -0.1 * (-1)**i
            smooth_motor_move(dance_targets, steps=5)

    elif action_index == 5:  # Happy Dance (Happy)
        # Full Happy Dance: swing arms, tilt body
        dance_targets = {}
        for i in range(8):
            dance_targets["LShoulderPitch"] = 0.5 * (-1)**i
            dance_targets["RShoulderPitch"] = 0.5 * (-1)**i
            dance_targets["LHipPitch"] = 0.2 * (-1)**i
            dance_targets["RHipPitch"] = 0.2 * (-1)**i
            dance_targets["HeadYaw"] = 0.3 * (-1)**i
            dance_targets["HeadPitch"] = -0.15 * (-1)**i
            smooth_motor_move(dance_targets, steps=5)

    elif action_index == 0:  # Patrol handled by do_patrol_step
        pass

    if action_index != 0:
        reset_pose(steps=4)

# ------------------------------
# EMOTION PREDICTION
# ------------------------------
def predict_emotion(face_img):
    """Predict emotion from face image using Keras model"""
    try:
        if face_img is None or face_img.size == 0:
            return "Neutral"
        img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        preds = emotion_model.predict(img, verbose=0)
        label_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = EMOTION_LABELS_7[label_idx]
        if confidence < CONFIDENCE_THRESHOLD:
            return "Neutral"
        return label
    except Exception as e:
        logging.exception("predict_emotion error: %s", e)
        return "Neutral"

# ------------------------------
# MEDIAPIPE + WEBCAM
# ------------------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# ------------------------------
# MAIN LOOP
# ------------------------------
frame_count = 0
last_label = "Neutral"
current_action_index = 0  # default Patrol

try:
    while robot.step(TIME_STEP) != -1:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        face_detected = False
        largest_area = 0
        best_crop = None
        best_coords = None

        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = max(0, int(box.xmin * w))
                y1 = max(0, int(box.ymin * h))
                x2 = min(w, int((box.xmin + box.width) * w))
                y2 = min(h, int((box.ymin + box.height) * h))
                if (x2 - x1) * (y2 - y1) > largest_area:
                    largest_area = (x2 - x1) * (y2 - y1)
                    best_crop = frame[y1:y2, x1:x2]
                    best_coords = (x1, y1, x2, y2)

            if best_crop is not None:
                face_detected = True
                if frame_count % FRAME_SKIP == 0:
                    label = predict_emotion(best_crop)
                    last_label = label
                    q_state = map_keras_to_q_state(label)
                    new_action = decide_optimal_action(q_state)

                    # Map Q-action index to NAO action
                    if label == "Happy":
                        new_action = 5  # Happy Dance
                    elif label == "Surprise":
                        new_action = 4  # Big Dance

                    if new_action != 0 and new_action != current_action_index:
                        execute_nao_action(new_action)
                        current_action_index = new_action
                    elif new_action == 0 and current_action_index != 0:
                        reset_pose(steps=4)
                        current_action_index = 0

                if best_coords:
                    x1, y1, x2, y2 = best_coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,0), 2)
                    cv2.putText(frame, last_label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

        # No face detected
        if not face_detected:
            if current_action_index != 0:
                reset_pose(steps=4)
                current_action_index = 0
            do_patrol_step()
            cv2.putText(frame, "NO FACE DETECTED (Patrol Mode)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("NAO Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Shutting down NAO controller.")
