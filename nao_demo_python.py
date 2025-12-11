"""
Module: LH/LM Intelligent Robotics (30227,30244)

Description: 
Final Controller for Coursework 2.
Integrated Face Detection, Q-Learning Decision, and Motor Control.
Note: Fixed the balance issue when resetting from Happy pose.
"""

import os
import cv2
import numpy as np
import csv
import time
import math
# Stop tensorflow logs, very annoying in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from controller import Robot
from tensorflow.keras.models import load_model
import mediapipe as mp

# ==============================================================================
# [PART 1] CONFIGURATION
# All parameters are here. Easy to tune.
# ==============================================================================
class Config:
    # --- System ---
    TIME_STEP = 32
    WEBCAM_ID = 0  # Use 0 for default laptop camera
    
    # --- Files (Check paths!) ---
    MODEL_PATH = "emotion_model_ver9.h5"
    Q_TABLE_PATH = "nao_emotion_qtable_tuned.npy"
    LOG_FILE = "mission_log_final.csv"
    
    # --- Thresholds ---
    CONF_THRESH = 0.6
    
    # --- Balance Parameters ---
    # [Fix]: Added this offset to prevent falling backwards when resetting.
    # A negative value means leaning forward slightly.
    LEAN_FORWARD_OFFSET = -0.1 
    
    # --- Logic ---
    # 0:Patrol, 1:Happy, 2:Angry, 3:Sad, 4:Surprise
    ACTIONS = ['Patrol', 'Raise Hands', 'Stomp', 'Shake Head', 'Big Dance']


# ==============================================================================
# [PART 2] NAO DRIVER (Hardware Interface)
# Handles motors and sensors only. No complex logic.
# ==============================================================================
class NaoDriver:
    def __init__(self):
        print("Driver: Initialising motors...")
        self.robot = Robot()
        self.motors = {}
        
        # Define joints we need to control
        # It's a long list...
        names = [
            "HeadYaw", "HeadPitch",
            "LShoulderPitch", "LShoulderRoll", "RShoulderPitch", "RShoulderRoll",
            "LHipPitch", "RHipPitch", "LKneePitch", "RKneePitch",
            "LAnklePitch", "RAnklePitch", "LHipRoll", "RHipRoll"
        ]
        
        for n in names:
            self.motors[n] = self.robot.getDevice(n)
            
        # Init Webcam
        # We use external webcam because Webots camera simulation is slow on my laptop
        self.cap = cv2.VideoCapture(Config.WEBCAM_ID)
        # Set low resolution for speed
        self.cap.set(3, 320)
        self.cap.set(4, 240)
        
        if not self.cap.isOpened():
            print("Error: Webcam not open!")

    def set_joints(self, target_dict):
        # Set motor positions directly
        for name, val in target_dict.items():
            if name in self.motors:
                self.motors[name].setPosition(val)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        return frame
        
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ==============================================================================
# [PART 3] NAO BRAIN (AI & Decision)
# Handles Emotion Recognition and Q-Learning.
# ==============================================================================
class NaoBrain:
    def __init__(self):
        print("Brain: Loading models...")
        
        # 1. Load Emotion Model (Keras)
        try:
            self.model = load_model(Config.MODEL_PATH)
            print(">>> Keras Model Loaded.")
        except: 
            print("Error: Keras model failed to load.")
            
        # 2. Load RL Policy (Q-Table)
        if os.path.exists(Config.Q_TABLE_PATH):
            self.q_table = np.load(Config.Q_TABLE_PATH)
            # Safe check for shape
            if self.q_table.shape != (5,5):
                print("Warning: Q-Table shape is wrong! Using zeros.")
                self.q_table = np.zeros((5,5))
            else:
                print(">>> Q-Table Loaded.")
        else:
            self.q_table = np.zeros((5,5))
            
        # 3. Face Detection
        self.mp_face = mp.solutions.face_detection.FaceDetection(0.5)
        
        # 4. Logger (For report data)
        self.f_log = open(Config.LOG_FILE, 'w', newline='')
        self.writer = csv.writer(self.f_log)
        self.writer.writerow(["Time", "Emotion", "Action", "Confidence"])

    def process_image(self, frame):
        # Mediapipe process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)
        
        if not results.detections: return None
        
        # Only take the first face
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x, y = int(bbox.xmin*w), int(bbox.ymin*h)
        bw, bh = int(bbox.width*w), int(bbox.height*h)
        
        # Check boundary
        if x<0 or y<0 or bw<1 or bh<1: return None
        
        # Draw for UI
        cv2.rectangle(frame, (x,y), (x+bw, y+bh), (0,255,0), 2)
        
        # Return cropped face
        return frame[y:y+bh, x:x+bw]

    def predict(self, face_img):
        # Resize to 96x96 for model
        rz = cv2.resize(face_img, (96, 96))
        norm = rz.astype("float32") / 255.0
        batch = np.expand_dims(norm, axis=0)
        
        # Predict
        preds = self.model.predict(batch, verbose=0)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        
        # Map 7 classes to 5 RL states
        labels_7 = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        raw_emo = labels_7[idx]
        
        # Mapping logic:
        # 0:Neutral, 1:Happy, 2:Angry, 3:Sad, 4:Surprise
        st_idx = 0
        if raw_emo == 'Happy': st_idx = 1
        elif raw_emo == 'Angry': st_idx = 2
        elif raw_emo == 'Sad': st_idx = 3
        elif raw_emo == 'Surprise': st_idx = 4
        # Disgust/Fear -> Neutral
        
        return raw_emo, st_idx, conf

    def get_action_from_q(self, st_idx):
        # Choose action with highest Q value
        return int(np.argmax(self.q_table[st_idx]))


# ==============================================================================
# [PART 4] MAIN CONTROLLER
# Connects Driver and Brain. Contains the Control Loop.
# ==============================================================================
def main():
    driver = NaoDriver()
    brain = NaoBrain()
    
    # Variables for state control
    current_act = 0
    t_start_act = 0
    is_acting = False
    
    print(">>> System Started. Press 'q' to exit.")
    
    while driver.robot.step(Config.TIME_STEP) != -1:
        # 1. Get visual input
        frame = driver.get_frame()
        if frame is None: continue
        
        face_img = brain.process_image(frame)
        
        # Default action is Patrol (0)
        act_idx = 0 
        
        # 2. Brain Logic (Only if not currently performing an action)
        if face_img is not None and not is_acting:
            try:
                emo, st_idx, conf = brain.predict(face_img)
                
                if conf > Config.CONF_THRESH:
                    act_idx = brain.get_action_from_q(st_idx)
                    
                    # Log data
                    brain.writer.writerow([time.time(), emo, Config.ACTIONS[act_idx], conf])
                    
                    # Display Text
                    cv2.putText(frame, f"{emo}->{Config.ACTIONS[act_idx]}", (10,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                               
                    # If action is not Patrol, lock the state
                    if act_idx != 0:
                        is_acting = True
                        t_start_act = driver.robot.getTime()
                        current_act = act_idx
                        print(f"Action Triggered: {Config.ACTIONS[act_idx]}")
            except: 
                pass

        # 3. Motion Controller (The Physics Part)
        
        # Define Neutral/Safe Pose first
        # [Important]: We add LEAN_FORWARD_OFFSET here to fix the falling back issue!
        safe_pose = {
            "LShoulderPitch": 1.6, "RShoulderPitch": 1.6,
            "HeadPitch": 0.0,
            # Crouch legs slightly
            "LKneePitch": 0.7, "RKneePitch": 0.7,
            # Move CoM forward
            "LHipPitch": -0.4 + Config.LEAN_FORWARD_OFFSET, 
            "RHipPitch": -0.4 + Config.LEAN_FORWARD_OFFSET,
            "LAnklePitch": -0.3, "RAnklePitch": -0.3
        }

        if is_acting:
            # Check timeout (Action duration 3s)
            if driver.robot.getTime() - t_start_act > 3.0:
                is_acting = False
                current_act = 0
                print("Action finished. Resetting...")
            
            # Execute specific action logic
            if current_act == 1: # Happy -> Hands Up
                # Need to lean forward more to balance raised hands
                driver.set_joints({
                    "LShoulderPitch": -1.5, "RShoulderPitch": -1.5,
                    "LShoulderRoll": 0.2, "RShoulderRoll": -0.2,
                    "HeadPitch": -0.3,
                    # Extra counter-balance
                    "LHipPitch": -0.6, "RHipPitch": -0.6
                })
                
            elif current_act == 2: # Angry -> Stomp
                driver.set_joints({
                    "HeadYaw": 0.3, "LHipPitch": -0.3, "RHipPitch": -0.3
                })
                
            elif current_act == 3: # Sad -> Shake Head
                t = driver.robot.getTime()
                shake = 0.5 * math.sin(t * 5.0)
                driver.set_joints({
                    "HeadPitch": 0.4, # Look down
                    "HeadYaw": shake,
                    "LShoulderPitch": 1.8, "RShoulderPitch": 1.8
                })
                
            elif current_act == 4: # Dance
                t = driver.robot.getTime()
                # Offset sine wave so arms don't hit body
                driver.set_joints({
                    "LShoulderRoll": 0.6 + 0.5 * math.sin(t*4),
                    "RShoulderRoll": -0.6 - 0.5 * math.sin(t*4)
                })
        else:
            # Idle Mode: Patrol (Arm swing) + Safe Pose
            t = driver.robot.getTime()
            swing = 0.3 * math.sin(t * 2.0)
            
            # Apply base pose first
            driver.set_joints(safe_pose)
            # Add arm swing on top
            driver.set_joints({
                "LShoulderPitch": 1.6 + swing, 
                "RShoulderPitch": 1.6 - swing
            })

        # Show UI
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'): break
        
    driver.close()

if __name__ == "__main__":
    main()
