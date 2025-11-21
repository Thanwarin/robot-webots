from controller import Supervisor, Keyboard, Motion
import cv2
import numpy as np

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# ===== Motion 文件路径 =====
MOTION_PATH = r"D:\nao_motions\motions\\"

# 基本移动动作
forward = Motion(MOTION_PATH + "Forwards.motion")
backward = Motion(MOTION_PATH + "Backwards.motion")
turn_left = Motion(MOTION_PATH + "TurnLeft40.motion")
turn_right = Motion(MOTION_PATH + "TurnRight40.motion")

# 情绪动作
happy_motion = Motion(MOTION_PATH + "HandWave.motion")
angry_motion = Motion(MOTION_PATH + "Shoot.motion")
sad_motion = Motion(MOTION_PATH + "Backwards.motion")
surprised_motion = Motion(MOTION_PATH + "TaiChi.motion")

current_motion = None  # 全局动作记录


# ===== 摄像头 CameraTop =====
camera = robot.getDevice("CameraTop")
camera.enable(timestep)

camera_width = camera.getWidth()
camera_height = camera.getHeight()
print(f"CameraTop Enabled: {camera_width} x {camera_height}")


# ===== 声呐 =====
sonar_left = robot.getDevice("Sonar/Left")
sonar_right = robot.getDevice("Sonar/Right")
sonar_left.enable(timestep)
sonar_right.enable(timestep)
SAFE_DISTANCE = 0.45   # 45cm 避障安全距离


def get_distance():
    """获取左/右声呐最小值"""
    return min(sonar_left.getValue(), sonar_right.getValue())


# ===== 键盘 =====
keyboard = Keyboard()
keyboard.enable(timestep)


# ===== 动作播放函数 =====
def play(m):
    """安全播放动作"""
    global current_motion
    try:
        if current_motion and current_motion != m:
            current_motion.stop()
    except:
        pass
    current_motion = m
    current_motion.play()


def stop_motion():
    """停止动作"""
    global current_motion
    if current_motion:
        try:
            current_motion.stop()
        except:
            pass
    current_motion = None


def play_emotion(m):
    """表情动作"""
    global current_motion
    try:
        if current_motion:
            current_motion.stop()
    except:
        pass
    current_motion = m
    current_motion.play()


print("Reagy!Go!")


# 主循环
while robot.step(timestep) != -1:

    
    # 0. 摄像头画面处理
    
    img = camera.getImage()
    if img:
        frame = np.frombuffer(img, np.uint8).reshape((camera_height, camera_width, 4))
        frame = frame[:, :, :3]  # 去除 alpha 通道
        cv2.imshow("NAO Camera", frame)
        cv2.waitKey(1)

   
    # 1. 键盘控制

    key = keyboard.getKey()

    # ---- 情绪动作 ----
    if key == ord('H'):
        print("😊 Happy emotion")
        play_emotion(happy_motion)

    elif key == ord('A'):
        print("😠 Angry emotion")
        play_emotion(angry_motion)

    elif key == ord('S'):
        print("😢 Sad emotion")
        play_emotion(sad_motion)

    elif key == ord('U'):
        print("😲 Surprised emotion")
        play_emotion(surprised_motion)

    # ---- 手动运动 ----
    if key != -1:
        if key == Keyboard.UP:
            if get_distance() > SAFE_DISTANCE:
                play(forward)
        elif key == Keyboard.DOWN:
            play(backward)
        elif key == Keyboard.LEFT:
            play(turn_left)
        elif key == Keyboard.RIGHT:
            play(turn_right)
        continue

    # 2. 自动行走

    dist = get_distance()

    if dist > SAFE_DISTANCE:
        play(forward)
        continue


    # 3. 自动避障

    print("⚠ 检测到障碍 → 停止")
    stop_motion()

    # ---- 原地小角度左转尝试避障 ----
    print("↪ 尝试小角度左转避障…")
    play(turn_left)
    robot.step(6 * timestep)

    if get_distance() > SAFE_DISTANCE:
        print("✓ 左侧已通畅 → 前进")
        continue

    # ---- 左侧仍然堵塞 → 原地更大角度右转 ----
    print("↪ 左侧依旧堵塞 → 尝试右转…")
    play(turn_right)
    robot.step(10 * timestep)

    if get_distance() > SAFE_DISTANCE:
        print("✓ 右侧已通畅 → 前进")
        continue

    # ---- 四周都有障碍 ----
    print("❌ 四周堵塞 → 停止等待")
    stop_motion()
