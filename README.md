# robot-webots

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
