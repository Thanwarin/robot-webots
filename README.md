# Design and Implementation of an Integrated Control Framework for a NAO Humanoid Robot in Webots
Facial Emotion Recognition for Human-Robot Interaction using FER/DeepFace and Webots Simulation.

## Overview

This repository contains code and simulation tools for enabling a NAO humanoid robot to recognize human facial emotions and respond with expressive, robot-like gestures inside the Webots simulator.

The system integrates:
- Facial Emotion Recognition (FER / DeepFace)
- Q-Learning–based action selection
- NAO expressive gesture execution
to create an interactive and responsive HRI pipeline.

## Features
- **Facial Emotion Recognition**
  - Supports FER-2013 model and DeepFace emotion classifier
  - Detects Neutral, Happy, Angry, Sad, Surprise
  - Real-time webcam input supported (15–20 FPS)

- **Emotion–Action Mapping**
  - Tabular Q-Learning maps detected emotions → robot gestures
  - Includes expressive animations such as:
    - Happy Dance
    - Surprise Gesture
    - Neutral, Sad, Angry responses

- **Webots Simulation**
  - Fully interactive NAO robot environment
  - Robot performs gestures based on live human emotion input
  - Supports real-time HRI experiments

- **Performance Metrics**
  - ~64% emotion recognition accuracy (FER-2013 validation)
  - ~90% correct response rate between detected emotion and Q-Learning action
  - 15–20 FPS live processing

## Getting Started
1. Clone this repository
    ```bash
    git clone https://github.com/Thanwarin/robot-webots.git

2. Install required dependencies
    ```bash
    pip install -r requirements.txt

3. Download pre-trained models

  Place the following files in the same directory as nao_demo_python.py:
  - Emotion Recognition Model: [Download here](https://drive.google.com/file/d/1b8FfnHOdwUhxmWSe27iqqs89xyuAc-bc/view?usp=sharing)
    - NAO Q-Table (Tuned): [Download here](https://drive.google.com/file/d/16vVjgt81T-OkOC9KZ-iL4VJEDAYYSj9R/view?usp=sharing)


4. Launch Webots and run the controller
    ```bash
    python nao_demo_python.py

## Expected Behavior

Once the program runs:
- The NAO robot detects your facial emotion through the webcam
- The Q-Learning policy selects an appropriate gesture
- NAO performs the gesture immediately

Example mappings:

Emotion	Gesture
Happy	Happy Dance
Surprise	Surprise Gesture
Neutral / Angry / Sad	Corresponding expressive actions
<p align="center"> <img width="450" alt="happy_dance_example" src="https://github.com/user-attachments/assets/7f2b1ae7-007d-4299-af1d-52d0995d295a" /> </p>

## Project Structure

├── emotion_detection_fer.ipynb          # FER model training/testing  

├── emotion_detection_DeepFace.ipynb     # DeepFace emotion detection demo  

├── nao_demo_python.py                   # Main controller (Webcam + FER + Q-Learning + NAO actions) 

├── worlds/                              # Webots NAO simulation world  

└── controllers/                         # Webots controller files  

## References
- Goodfellow et al., Challenges in Representation Learning, ICML 2013
- Sutton & Barto, Reinforcement Learning: An Introduction, MIT Press, 2018
- Aldebaran / SoftBank Robotics, NAO Humanoid Robot Documentation
- O. Michel, Webots: Professional Mobile Robot Simulation, 2004

## Acknowledgements
- FER-2013 dataset
- DeepFace library
- Webots / NAO simulation tools
