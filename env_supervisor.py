
from controller import Supervisor
import sys
print(">>> Webots Python Path:", sys.executable)

import cv2   # 这一行现在会报错没关系
import numpy as np


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# ===== 获取 NAO 节点 =====
nao = robot.getFromDef("NAO")
translation_field = nao.getField("translation")
rotation_field = nao.getField("rotation")

# ===== 获取摄像头设备 =====
camera = robot.getDevice("CameraTop")
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()
print("CameraTop Enabled! Resolution:", width, "x", height)

# ==== 获取避障超声波传感器 ====
sonar_left = robot.getDevice("Sonar/Left")
sonar_right = robot.getDevice("Sonar/Right")
sonar_left.enable(timestep)
sonar_right.enable(timestep)

def get_distance():
    l = sonar_left.getValue()
    r = sonar_right.getValue()
    return min(l, r)
    
# ===== 移动函数 =====
def move_forward(step=0.01):
    pos = translation_field.getSFVec3f()
    pos[0] += step  # 控制 X 方向移动
    translation_field.setSFVec3f(pos)

while robot.step(timestep) != -1:

    # ==== 读取摄像头画面 ====
    img = camera.getImage()
    if img is not None:
        frame = np.frombuffer(img, np.uint8).reshape((height, width, 4))
        frame = frame[:, :, :3]  # 移除 alpha 通道
        cv2.imshow("NAO CameraTop", frame)
        cv2.waitKey(1)

    # ==== 获取超声波测距 ====
    dist = get_distance()
    print("Sonar distance:", dist)

    SAFE_DISTANCE = 0.45  # 小于 45cm 则视为障碍物

    # ==== 避障判断 ====
    if dist < SAFE_DISTANCE:
        print("⚠ Obstacle detected! Distance:", dist)
        # 可以原地不动/后退/转弯，这里先停止:
        # TODO：后续你可以在这里加入 emotion-based 行为
    else:
        move_forward(0.005)
        print("NAO moving & capturing...")


