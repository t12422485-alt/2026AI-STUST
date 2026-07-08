# 02 — YOLO11n 預訓練模型即時 webcam 偵測
#
# Ultralytics 官方的 yolo11n.pt 是在 COCO 80 類上訓練的（同 DAY9 MediaPipe）。
# 我們直接拿來玩，第一次執行會自動下載 ~5.5MB。
#
# 需要：pip install ultralytics opencv-python pillow

import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ====== 載入模型 ======
# 第一次執行會自動下載到當前目錄
model = YOLO("yolo11n.pt")
print(f"模型類別數：{len(model.names)}")

# ====== 中文字型（cv2.putText 不支援 CJK）======
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

# COCO 80 類的中文翻譯（同 DAY9 08）
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
    bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    draw.rectangle((x, y, x + tw + 6, y + th + 6), fill=bg_rgb)
    draw.text((x + 3, y + 3), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ====== 開 webcam ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("無法開啟 webcam")

print("按 q 離開")

fps_hist = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t0 = time.time()

    # Ultralytics 一行推論；imgsz 越小越快，320 對即時應用很夠
    results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)

    # 解析結果（Ultralytics 已幫我們做完 NMS）
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            name_en = model.names[cls_id]
            name_zh = COCO_ZH.get(name_en, name_en)
            color   = 依類別配色(name_en)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            frame = 畫中文標籤(frame, f"{name_zh} {conf*100:.0f}%",
                              (int(x1), max(0, int(y1) - 30)), 20, color)

    # FPS
    fps_hist.append(1.0 / max(time.time() - t0, 1e-3))
    if len(fps_hist) > 30:
        fps_hist.pop(0)
    fps = sum(fps_hist) / len(fps_hist)
    frame = 畫中文標籤(frame, f"FPS {fps:.1f}", (10, 10), 22, (30, 30, 30))

    cv2.imshow("YOLO11n (預訓練 COCO 80 類)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
