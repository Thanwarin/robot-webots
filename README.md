# Design and Implementation of an Integrated Recognition and Control Framework for the NAO Humanoid Robot in Webots

Facial Emotion Recognition and Q-Learning–based Action Selection for Human–Robot Interaction using Webots Simulation.



## Overview

This project integrates **Facial Emotion Recognition (FER/DeepFace)** with a **Q-Learning action-selection model** to enable a NAO humanoid robot inside Webots to:

- Recognize human facial emotions using a webcam  
- Interpret emotion → action using a Q-table  
- Perform expressive animations with improved stability  
- Execute actions only after completing the previous one (balance-safe behavior)  

The final system achieves smooth, autonomous human–robot interaction (HRI) inside Webots.


## Features

### Facial Emotion Recognition
- Supports FER-2013 and DeepFace models  
- Detects:
  - Happy  
  - Sad  
  - Angry  
  - Surprise  
  - Neutral  
- Real-time webcam processing at **15–20 FPS**



### Emotion–Action Mapping (Q-Learning)

The Q-Learning model maps each recognized emotion to a specific robot gesture.

| Emotion             | Gesture                                                         |
|--------------------|------------------------------------------------------------------|
| **Happy**          | Raise both hands and look up                                     |
| **Sad**            | Lower head and shake head                                        |
| **Angry**          | Stomp feet forcefully                                            |
| **Surprise**       | Lift head with shocked posture / arms up                         |
| **Neutral/Scared** | Swing arms in place like a walking gesture                       |

Improvements:
- Removed unused **disgust** category  
- Ensured NAO **finishes each gesture** before detecting the next emotion  
- Adjusted center of gravity for improved stability  



### Stability Optimization

To prevent NAO from losing balance:
- Added action-complete timing control  
- Optimized transitions between gestures  
- Smoothed reset-to-neutral phases

## Expected Robot Behavior

After the system starts:

- Webcam detects the user’s facial emotion  
- Q-Learning selects the appropriate robot gesture  
- NAO performs the full gesture smoothly  
- Next detection begins only after finishing the previous gesture  




## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Thanwarin/robot-webots.git
```
### 2. Install required Python dependencies
```bash
pip install -r requirements.txt
```
### 3. Download pretrained models

Place these files in the same directory as nao_demo_python.py:
  - Emotion Recognition Model: [Download here](https://drive.google.com/file/d/1b8FfnHOdwUhxmWSe27iqqs89xyuAc-bc/view?usp=sharing)
  - NAO Q-Table (Tuned): [Download here](https://drive.google.com/file/d/16vVjgt81T-OkOC9KZ-iL4VJEDAYYSj9R/view?usp=sharing)

### 4. Launch Webots with the controller

Set the controller of the NAO robot to:

```bash
nao_demo_python.py
```


Then run Webots or execute:
```bash
python nao_demo_python.py
```

### Example Gesture Image
<p align="center">
  <img width="450" src="https://github.com/user-attachments/assets/7f2b1ae7-007d-4299-af1d-52d0995d295a" alt="happy action"/>
</p>



## Robot Behavior & Gesture Mapping

When the system is running, the NAO robot detects the user's facial emotion through the webcam and selects the appropriate gesture according to the Q-Learning policy. Each gesture is completed fully before the robot detects the next emotion.

| Emotion             | Gesture                                                         | Example Image |
|--------------------|-----------------------------------------------------------------|---------------|
| **Happy**          | Raise both hands and look up (Happy Dance)                      | <img width="300" height="446" alt="image" src="https://github.com/user-attachments/assets/dd299bc7-c12a-44c1-821d-59a393651b33" />
| **Sad**            | Lower head and shake head                                        | <img width="300" src="https://github.com/Thanwarin/robot-webots/blob/tmp/tmp/Part%202%20Chenran%20Shi/images/Sad.jpg" alt="Sad"/> |
| **Angry**          | Stomp feet forcefully                                            | <img width="300" src="https://github.com/Thanwarin/robot-webots/blob/tmp/tmp/Part%202%20Chenran%20Shi/images/Angry.jpg" alt="Angry"/> |
| **Surprise**       | Lift head with shocked posture / move arms and legs like surprised | <img width="300" src="https://github.com/Thanwarin/robot-webots/blob/tmp/tmp/Part%202%20Chenran%20Shi/images/Surprise.jpg" alt="Surprise"/> |
| **Neutral / Scared** | Swing arms in place like a walking gesture                      | <img width="300" height="421" alt="image" src="https://github.com/user-attachments/assets/4bf696cd-fcce-431a-afe5-09d3f38f4ffc" />|

**Key Improvements:**
- NAO finishes each gesture before detecting the next emotion  
- Adjusted center of gravity for improved stability  
- Smooth transitions between gestures and reset-to-neutral phases


## Project Structure

├── emotion_detection_fer.ipynb          # FER model training/testing (file)

├── emotion_detection_DeepFace.ipynb     # DeepFace emotion detection demo (file)

├── nao_demo_python.py                   # Main controller (file)

├── worlds/                              # Webots NAO simulation world (folder)

└── controllers/                         # Webots controller files (folder)
## System Improvements (Before vs After Update)

### Before Updating
- Vision model recognized 6 emotions, but robot performed only 1 gesture  
- Robot frequently interrupted previous actions → unstable behavior  
- Original Q-table: 5×5 but 6 emotion categories  

### After Updating
- Unified emotion categories (removed **disgust**)  
- Rebuilt Q-table with correct number of states/actions  
- Robot completes each action before detecting a new emotion  
- Improved posture transitions and gesture fluidity  


 
## Contributors

- **Thanwarin Luangmanotham**: Develop contrastive loss function, pre-training/fine-tuning pipeline, and human pose–object fusion module for social representation learning; implement parts of robot policy integration to test learned representations.
- **Shuqin Wang**: Build multi-dataset processing pipeline with augmentation and quality evaluation for social behavior data; implement data feeding modules for robot experiments.
- **Chenran Shi**: Implement Policy Network, social rule mapping, and robot action safety verification for decision-control; program robot to execute learned social behaviors in simulated scenarios.
- **Yintao Ma**: Develop an evaluation system with quantitative and qualitative metrics, experiment automation, and ethical protocols; implement test scripts that directly interact with robot policies to generate evaluation data.


---

## Acknowledgements

- FER-2013 Dataset  
- DeepFace Library  
- Webots Simulation Tools  
- NAO Humanoid Robot

---

## References

- Abadi, M. et al., *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems*, 2015.
- Aldebaran Robotics, *NAO Humanoid Robot Documentation*, SoftBank Robotics, 2014.
- Borenstein, J. & Koren, Y., *Real-Time Obstacle Avoidance for Fast Mobile Robots*, IEEE Transactions on Systems, Man, and Cybernetics, vol. 19, no. 5, pp. 1179–1187, 1989.
- Brooks, R. A., *A robust layered control system for a mobile robot*, IEEE Journal on Robotics and Automation, vol. 2, no. 1, pp. 14–23, 1986.
- Breazeal, C., *Emotion and Sociable Humanoid Robots*, International Journal of Human Computer Studies, vol. 59, no. 1-2, pp. 119–155, 2003.
- Da Silva, G. & Melo, F., *DeepFace: Facial recognition with deep learning*, in 2018 International Conference on Robotics and Automation, 2018, pp. 1–6.
- Goodfellow, I. et al., *Challenges in Representation Learning: A Report on Three Machine Learning Contests*, ICML 2013 Workshop, 2013.
- Guzzi, J., Cully, A., & Mouret, J. B., *Learning the behavior of a mobile robot using reinforcement learning in human-robot interaction*, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2013, pp. 2322–2327.
- Lugaresi, L. et al., *MediaPipe: A Framework for Building Perception Pipelines*, arXiv:1906.08172, 2019.
- Michel, P., Rohmer, E., & Singh, S. P., *Webots: Professional Mobile Robot Simulation*, International Journal of Advanced Robotic Systems, 2004.
- Rohmer, E., Singh, S. P., & Freese, M., *Webots: a robot simulator for teaching and research*, International Journal of Advanced Robotic Systems, vol. 10, no. 3, pp. 1–10, 2013.
- Shao, W., Zhang, L., & He, J., *Reinforcement learning for adaptive human-robot interaction*, Robotics and Autonomous Systems, vol. 124, pp. 103395, 2020.
- Sutton, R. S. & Barto, A. G., *Reinforcement Learning: An Introduction*, 2nd Edition, MIT Press, 2018.
- Tapus, A., Mataric, M. J., & Scassellati, B., *Socially assistive robotics [Grand challenges of robotics]*, IEEE Robotics & Automation Magazine, vol. 14, no. 1, pp. 35–42, 2007.


