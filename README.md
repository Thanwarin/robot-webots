# Design and Implementation of an Integrated Control Framework for a NAO Humanoid Robot in Webots

Facial Emotion Recognition for Human-Robot Interaction using FER/DeepFace and Webots simulation.

## Overview
This repository contains code and simulation environments for enabling a NAO humanoid robot to recognize human facial emotions and respond with appropriate actions in Webots. The system integrates facial emotion recognition models with a Q-Learning policy to select expressive robot gestures based on detected human emotions.

## Features
- **Facial Emotion Recognition:** 
  - Using FER-2013 dataset and DeepFace for detecting basic emotions (Neutral, Happy, Angry, Sad, Surprise).
  - Supports real-time webcam input.
- **Emotion-Action Mapping:** 
  - Tabular Q-Learning maps detected emotions to robot actions.
  - Actions include gestures such as Happy Dance and Surprise reactions.
- **Webots Simulation:** 
  - Simulate NAO robot performing gestures in response to human emotions.
  - Real-time interaction with human participants.
- **Performance Metrics:**
  - Emotion recognition accuracy: ~64% on FER-2013 validation.
  - Correct response rate: ~90% of detected emotions resulted in intended Q-Learning action.
  - Processing speed: 15–20 FPS.

## Getting Started
1. Clone this repository:
    ```bash
    git clone https://github.com/Thanwarin/robot-webots.git

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt

3. Download pre-trained models and place them in the project directory (same folder as nao_demo_python.py):
- Emotion Recognition Model: [Download here
](https://drive.google.com/file/d/1b8FfnHOdwUhxmWSe27iqqs89xyuAc-bc/view?usp=sharing)
- NAO Q-Table (Tuned): [Download here](https://drive.google.com/file/d/16vVjgt81T-OkOC9KZ-iL4VJEDAYYSj9R/view?usp=sharing)

4.Launch Webots simulation and run the emotion recognition script:
    ```bash
    python nao_demo_python.py

## Expected Behavior:
- The NAO robot detects emotions from the webcam in real-time.
- Performs corresponding gestures based on the Q-Learning action policy:
- Happy → Happy Dance
- Surprise → Big Surprise Gesture
- Neutral, Angry, Sad → Corresponding gestures
<img width="508" height="457" alt="happy_dance_example" src="https://github.com/user-attachments/assets/7f2b1ae7-007d-4299-af1d-52d0995d295a" />


## Project Structure
- emotion_detection_fer.ipynb: Training and testing FER model.
- emotion_detection_DeepFace.ipynb: Using DeepFace for emotion recognition.
- nao_demo_python.py: Main controller integrating webcam input, emotion recognition, and Q-Learning action selection.
- Webots world files: Contain NAO robot setup for simulation.

## References
- Goodfellow et al., Challenges in Representation Learning: A Report on Three Machine Learning Contests, ICML 2013 Workshop.
- R.S. Sutton & A.G. Barto, Reinforcement Learning: An Introduction, 2nd Edition, MIT Press, 2018.
- Aldebaran Robotics, NAO Humanoid Robot Documentation, SoftBank Robotics, 2014.
- O. Michel, Webots: Professional Mobile Robot Simulation, International Journal of Advanced Robotic Systems, 2004.

## Acknowledgements
- FER-2013 dataset for facial emotion recognition.
- Webots and NAO robot simulation resources.

