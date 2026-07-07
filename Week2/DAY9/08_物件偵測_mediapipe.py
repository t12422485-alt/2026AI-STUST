# 08 — MediaPipe 物件偵測 (Object Detection)
#
# 相比 05_遷移學習_MobileNet 只回答「這張圖是什麼類別」，
# Object Detection 進階回答「東西在哪裡」+ 畫框框 + 一次抓多個。
#
# 用 Google 官方推薦的 EfficientDet-Lite0 模型（COCO 資料集，80 類），
# 320x320 輸入，第一次執行會自動下載 4.5MB 的 .tflite 檔。
#
# 需要：pip install mediapipe opencv-python pillow
# 若沒裝：先 pip install -r requirements.txt

import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

# ====== 模型下載 ======
BASE = Path(__file__).parent
MODEL_PATH = BASE / "efficientdet_lite0.tflite"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite")

if not MODEL_PATH.exists():
    print(f"第一次執行：下載 EfficientDet-Lite0 到 {MODEL_PATH} (~4.5MB) ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("下載完成！")

# ====== 中文字型 ======
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

# COCO 80 類的中文翻譯（沒翻的顯示英文原文）
COCO_ZH = {
    "person":"人", "bicycle":"腳踏車", "car":"汽車", "motorcycle":"機車",
    "airplane":"飛機", "bus":"公車", "train":"火車", "truck":"卡車",
    "boat":"船", "traffic light":"紅綠燈", "fire hydrant":"消防栓",
    "stop sign":"停止標誌", "parking meter":"停車計時器", "bench":"長椅",
    "bird":"鳥", "cat":"貓", "dog":"狗", "horse":"馬", "sheep":"羊",
    "cow":"牛", "elephant":"大象", "bear":"熊", "zebra":"斑馬",
    "giraffe":"長頸鹿", "backpack":"背包", "umbrella":"傘",
    "handbag":"手提包", "tie":"領帶", "suitcase":"行李箱", "frisbee":"飛盤",
    "skis":"滑雪板", "snowboard":"雪橇", "sports ball":"球", "kite":"風箏",
    "baseball bat":"球棒", "baseball glove":"棒球手套", "skateboard":"滑板",
    "surfboard":"衝浪板", "tennis racket":"網球拍", "bottle":"瓶子",
    "wine glass":"酒杯", "cup":"杯子", "fork":"叉子", "knife":"刀",
    "spoon":"湯匙", "bowl":"碗", "banana":"香蕉", "apple":"蘋果",
    "sandwich":"三明治", "orange":"橘子", "broccoli":"青花菜",
    "carrot":"紅蘿蔔", "hot dog":"熱狗", "pizza":"披薩", "donut":"甜甜圈",
    "cake":"蛋糕", "chair":"椅子", "couch":"沙發", "potted plant":"盆栽",
    "bed":"床", "dining table":"餐桌", "toilet":"馬桶", "tv":"電視",
    "laptop":"筆電", "mouse":"滑鼠", "remote":"遙控器", "keyboard":"鍵盤",
    "cell phone":"手機", "microwave":"微波爐", "oven":"烤箱",
    "toaster":"烤麵包機", "sink":"水槽", "refrigerator":"冰箱", "book":"書",
    "clock":"時鐘", "vase":"花瓶", "scissors":"剪刀", "teddy bear":"泰迪熊",
    "hair drier":"吹風機", "toothbrush":"牙刷",
}


def 依類別配色(name):
    """相同類別給相同顏色，用 hash 分配（BGR）"""
    h = hash(name) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def 畫中文標籤(img_bgr, text, xy, size, bg_bgr):
    """在指定位置畫「彩色底 + 白字」"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, size)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    x, y = xy
    # PIL 要 RGB，把 BGR 換過來
    bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    draw.rectangle((x, y, x + tw + 6, y + th + 6), fill=bg_rgb)
    draw.text((x + 3, y + 3), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ====== 建立 detector ======
options = vision.ObjectDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,   # 影片模式，同步處理
    max_results=10,                          # 一次最多回 10 個框
    score_threshold=0.4,                     # 信心低於 0.4 的丟掉
)
detector = vision.ObjectDetector.create_from_options(options)

# ====== 開 webcam ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("無法開啟 webcam")

print("按 q 離開；把物體對著鏡頭讓它畫框")

start_ts = time.time()
fps_hist = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()

    # MediaPipe 需要 RGB numpy 影像
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # VIDEO 模式需要單調遞增的時間戳（毫秒）
    ts_ms = int((time.time() - start_ts) * 1000)
    result = detector.detect_for_video(mp_image, ts_ms)

    # 畫每個偵測到的物件
    for det in result.detections:
        bbox = det.bounding_box
        cat  = det.categories[0]
        x, y = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height

        name_zh = COCO_ZH.get(cat.category_name, cat.category_name)
        color   = 依類別配色(cat.category_name)

        # 框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        # 上方彩色標籤
        frame = 畫中文標籤(frame, f"{name_zh} {cat.score*100:.0f}%",
                          (x, max(0, y - 30)), 20, color)

    # FPS
    fps_hist.append(1.0 / max(time.time() - frame_start, 1e-3))
    if len(fps_hist) > 30:
        fps_hist.pop(0)
    fps = sum(fps_hist) / len(fps_hist)
    frame = 畫中文標籤(frame, f"FPS {fps:.1f}", (10, 10), 22, (30, 30, 30))

    cv2.imshow("MediaPipe Object Detection (COCO 80 類)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
