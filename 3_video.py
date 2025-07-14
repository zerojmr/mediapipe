import time
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    GestureRecognizerResult,
)

# 顶部定义
idle_color   = (0, 255, 0)
active_color = (0, 0, 255)

# Tasks API 引入
BaseOptions       = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing 工具
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

# 缓存最新关键点结果
latest_result = None

# 方块样式及中心
square_size      = 100
square_color     = (0, 255, 0)
square_thickness = 2
box_center_x     = None
box_center_y     = None

# 捏合触发阈值（归一化距离）
pinch_thresh = 0.05

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# 创建 GestureRecognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)
recognizer = GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

last_ts = 0
min_interval = 0.1  # 100ms 调一次识别

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 镜像并拿到尺寸
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # 首次初始化方块中心
    if box_center_x is None:
        box_center_x = w // 2
        box_center_y = h // 2

    # 限频调用识别
    now = time.time()
    if now - last_ts > min_interval:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        recognizer.recognize_async(mp_image, int(now * 1000))
        last_ts = now

    # 复制一帧用于绘制
    annotated = frame.copy()

    is_grabbing = False

    # 如果有检测结果，尝试捏合拖拽
    if latest_result and latest_result.hand_landmarks:
        hand = latest_result.hand_landmarks[0]
        idx   = hand[8]
        thumb = hand[4]
        px, py = int(idx.x * w), int(idx.y * h)
        dx, dy = idx.x - thumb.x, idx.y - thumb.y

        x1 = box_center_x - square_size//2
        y1 = box_center_y - square_size//2
        x2 = x1 + square_size
        y2 = y1 + square_size

        if x1 < px < x2 and y1 < py < y2 and (dx*dx + dy*dy)**0.5 < pinch_thresh:
            box_center_x = px
            box_center_y = py
            is_grabbing = True

        # （可选）画手关键点
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand
        ])
        mp_drawing.draw_landmarks(
            annotated, proto,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style()
        )

    # 画方块
    x1 = box_center_x - square_size//2
    y1 = box_center_y - square_size//2
    x2 = x1 + square_size
    y2 = y1 + square_size

    color = active_color if is_grabbing else idle_color
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, square_thickness)

    # 显示并处理按键
    cv2.imshow('Gesture Drag Box', annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        box_center_x, box_center_y = w // 2, h // 2

# 循环外统一释放资源
recognizer.close()
cap.release()
cv2.destroyAllWindows()
