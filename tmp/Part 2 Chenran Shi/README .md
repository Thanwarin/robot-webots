  This was the second version of the README for the Q-Learning model training. The reason was that the Keras/TensorFlow versions inside the stable_baselines3 library were too old. At that time, the RL brain model was still trained based on my teammate Thanwarin Luangmanotham’s vision model emotion_model_ver4, and we tried to load it into the Webots NAO robot so the robot could read the face and choose the right action. But in the end, it still could not work together with her final vision model, emotion_model_ver9.

  So we changed to a Q-Learning method using a simple Q-table. This Q-Learning model was built by my teammate Thanwarin Luangmanotham, and she used a Webots controller named nao_demo_python.

  However, when I ran her Webots controller again, I found that the NAO robot still could not do different actions for different emotions. Because of this, I rebuilt and upgraded her controller. After this update, we got the final Webots controller for our whole project, named nao_demo_python final version.

# Design and Implementation of an Integrated Recognition and Control Algorithm for the NAO Humanoid Robot

Emotion-to-Action Conversion Using a Q-Learning Model and Webots NAO Robot Motion Simulation Algorithms

## Overview
Based on my teammate Thanwarin Luangmanotham’s final vision model and the Q-Learning training model, I upgraded the Q-Learning model by rewriting its code inside the Webots controller. After this, it could work together with her vision model. With this link, the NAO robot used the Q-table to judge and find the response value by itself. In the end, the NAO robot could recognize human facial expressions on its own and do the correct action.

## Parameter and Code Corrections

- **Before Updating and Reconstructing the Original Model Code:** 

  - After loading my teammate’s original model code, the vision model could recognize six emotions (happy, fear, sad, disgust, surprise, angry). But after the robot recognized these six emotions, it could only do one body-swaying action, and then it directly entered the patrol mode.
  
  - When running my teammate’s original code, the robot made the same action too often after detecting any emotion. Many times, the previous action had not finished yet, and the next action started right away. This made the robot unable to stand steadily, broke its balance, and caused the robot to fail to complete the full process.
  
- **After Updating and Reconstructing the Original Model Code:**
  
  - The original model code used a 5×5 Q-table, but there were six emotions. Also, the original code wrote that if the model could not detect an emotion, the robot would do the same action. The vision model could not detect the sixth emotion, so the NAO robot always received the same action signal. Based on this, I removed the sixth emotion (disgust) and improved the robot’s action after recognition.
  - To solve the problem of the NAO robot reacting too fast and losing balance, I added a rule in my Webots controller code. After the robot detects a facial expression, it must finish the full action process first. After the action is done, it will detect the next facial expression and then start the next full action process. This helps reduce the error where the robot starts a new action before the previous one stops because the recognition is too fast.I also adjusted the robot’s center of gravity when doing the happy action (raising both arms) and when the arms return to the reset position. This makes the robot more stable and helps prevent it from losing balance while doing the actions.
  
## Features

- **Optimizing the NAO Robot’s Actions After Emotion Recognition:** 
  - happy → raise both hands and look up
  - sad → lower the head and shake the head
  - angry → stomp the feet like showing anger
  - surprised → lift the head with a shocked face and stay still / or move arms and legs like being very surprised
  - neutral / scared → swing the arms in place to imitate a human walking gesture.
- **Optimized Action Posture of the NAO Robot in the Full Process:** 








































