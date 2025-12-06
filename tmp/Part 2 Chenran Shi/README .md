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
  - By adjusting the NAO robot’s posture after it finishes the correct action for the recognized expression, the robot can return to a stable standing position. This makes it easier for the robot to stay balanced and perform the next action smoothly.

## Start the Operation now.

1. Find and clone the code in my repository（but do not clone any files whose names end with “not”).
    ```bash
    https://github.com/Thanwarin/robot-webots/tree/tmp/tmp/Part%202%20Chenran%20Shi

2. Install the required Python libraries.
   
    [Download here](https://github.com/Thanwarin/robot-webots/blob/tmp/tmp/Part%202%20Chenran%20Shi/All%20required%20Python%20libraries.txt)

3. Put my teammate Thanwarin Luangmanotham’s vision model and the Q-Learning training model into a new folder under the Webots NAO robot controllers directory.
The folder name should be the same as nao_demo_python final version.py, and inside this folder you must include the file nao_demo_python final version.py.

    - Emotion Recognition Model: [Download here](https://drive.google.com/file/d/1b8FfnHOdwUhxmWSe27iqqs89xyuAc-bc/view?usp=sharing)
    - NAO Q-Table (Tuned): [Download here](https://drive.google.com/file/d/16vVjgt81T-OkOC9KZ-iL4VJEDAYYSj9R/view?usp=sharing)
    
4.Run Webots。
    - After running Webots, open the NAO robot scene. In the left program tree in Webots, find the NAO robot and click to open it. Then find the controller field and change it from nao to nao_demo_python final version.py. After that, save the world and run it.

## Expected Behavior of the NAO Robot:
- happy → raise both hands and look up
！[Happy](./images/Angry.jpg)
- sad → lower the head and shake the head
！[Sad](./images/Sad.jpg) 
- angry → stomp the feet like showing anger
！[Angry](./images/Angry.jpg)  
- surprised → lift the head with a shocked face and stay still / or move arms and legs like being very surprised
！[Surprise](./images/Surprise.jpg) 
- neutral / scared → swing the arms in place to imitate a human walking gesture.

## Project Structure
- Webots world files: Contain NAO robot setup for simulation.
- nao_demo_python final version.py: Main controller integrating webcam input, emotion recognition, and Q-Learning action selection.
- emotion_model_ver9.h5: Teammate Thanwarin’s vision recognition model helps the NAO robot recognize facial expressions.
- nao_emotion_qtable_tuned.npy：Teammate Thanwarin’s Q-table file is used by the Q-Learning model. Based on this Q-table, the model maps the emotion recognized by the robot to the correct area in the table, allowing the NAO robot to judge by itself and choose the right action.

## References
- Aldebaran Robotics, NAO Humanoid Robot Documentation, SoftBank Robotics, 2014.
- O. Michel, Webots: Professional Mobile Robot Simulation, International Journal of Advanced Robotic Systems, 2004.
- R.S. Sutton & A.G. Barto, Reinforcement Learning: An Introduction, 2nd Edition, MIT Press, 2018.















