# This was the second version of the README for the Q-Learning model training. The reason was that the Keras/TensorFlow versions inside the stable_baselines3 library were too old. At that time, the RL brain model was still trained based on my teammate Thanwarin Luangmanotham’s vision model emotion_model_ver4, and we tried to load it into the Webots NAO robot so the robot could read the face and choose the right action. But in the end, it still could not work together with her final vision model, emotion_model_ver9.

# So we changed to a Q-Learning method using a simple Q-table. This Q-Learning model was built by my teammate Thanwarin Luangmanotham, and she used a Webots controller named nao_demo_python.

# However, when I ran her Webots controller again, I found that the NAO robot still could not do different actions for different emotions. Because of this, I rebuilt and upgraded her controller. After this update, we got the final Webots controller for our whole project, named nao_demo_python final version.

# Design and Implementation of an Integrated Recognition and Control Algorithm for the NAO Humanoid Robot

Emotion-to-Action Conversion Using a Q-Learning Model and Webots NAO Robot Motion Simulation Algorithms

## Overview












































