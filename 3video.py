# 处理摄像头手势
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 手动定义 21 个关键点之间的连接关系
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# 初始化 GestureRecognizer（VIDEO 模式）
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,                            # ← 这里改成 2，就能检测并返回两只手的结果 :contentReference[oaicite:0]{index=0}
    min_hand_detection_confidence=0.5,      # 可选：手掌检测门槛
    min_hand_presence_confidence=0.5,       # 可选：关键点跟踪门槛
    min_tracking_confidence=0.5             # 可选：跟踪失败后是否重新检测
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")
start_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int((time.time() - start_time) * 1000)
    result = recognizer.recognize_for_video(mp_image, ts)

    h, w, _ = frame.shape
    # 画关键点
    if result.hand_landmarks:
        for lm_list in result.hand_landmarks:
            for lm in lm_list:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0,255,0), -1)
    # 画骨架连线
    if result.hand_landmarks:
        for lm_list in result.hand_landmarks:
            for i0, i1 in HAND_CONNECTIONS:
                p0 = lm_list[i0]; p1 = lm_list[i1]
                x0, y0 = int(p0.x * w), int(p0.y * h)
                x1, y1 = int(p1.x * w), int(p1.y * h)
                cv2.line(frame, (x0, y0), (x1, y1), (255,0,0), 2)
    # 写手势文字
    if result.gestures:
        for idx, g_list in enumerate(result.gestures):
            if g_list:
                g = g_list[0]
                txt = f"Hand {idx+1}: {g.category_name} {g.score:.2f}"
                cv2.putText(frame, txt, (10, 30+30*idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
