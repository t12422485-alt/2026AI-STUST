# 06 — 作業：接你自己訓練的 Teachable Machine 模型，做一個有實際用途的應用
#
# TODO 1) 到 Teachable Machine 訓練你自己的模型（見 02 教學）
#         想一個有意義的分類任務，例如：
#           - 口罩：戴 / 沒戴 / 戴錯位置
#           - 姿勢：坐姿正 / 駝背 / 站立
#           - 手勢：剪刀 / 石頭 / 布（拳皇對戰？）
#           - 情緒：微笑 / 皺眉 / 中性
#           - 桌面整齊度：乾淨 / 亂
#
# TODO 2) 匯出 → 把 keras_model.h5 + labels.txt 放到本資料夾
#
# TODO 3) 在下方 [自訂邏輯] 區段加上你的創意應用，例如：
#         - 分類結果變化時發聲（winsound / playsound）
#         - 記錄每次分類到 SQLite（結合 DAY6 web_basics）
#         - 觸發 GPIO / 開燈 / 顯示到別的視窗
#         - 分類是「駝背」超過 5 秒 → 跳警告視窗
#
# TODO 4) 交作業時：附上你的 model + 一段 30 秒的 demo 影片
#
# 需要：pip install tensorflow opencv-python numpy

import os
import time
import cv2
import numpy as np
import tensorflow as tf

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "keras_model.h5")
LABELS_PATH = os.path.join(BASE, "labels.txt")

# ====== 讀模型 + labels ======
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("無法開啟 webcam")

# ====== 狀態變數：追蹤上一次的類別和累積時間 ======
上次類別   = None
上次時間   = time.time()
持續秒數   = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 預處理
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    目前類別 = labels[top_idx]
    信心    = float(preds[top_idx])

    # ====== 累積目前類別持續多久 ======
    now = time.time()
    dt = now - 上次時間
    上次時間 = now
    if 目前類別 == 上次類別:
        持續秒數 += dt
    else:
        持續秒數 = 0.0
        上次類別 = 目前類別

    # ====== [自訂邏輯] 在這裡加你自己的動作！======
    # 範例：某個類別持續 3 秒以上就印警告
    # if 目前類別 == "駝背" and 持續秒數 > 3.0:
    #     print(">>> 警告：駝背太久了！")
    #
    # 範例：偵測到特定類別發聲（Windows）
    # if 目前類別 == "剪刀" and 持續秒數 > 1.0:
    #     import winsound
    #     winsound.Beep(1000, 100)

    # ====== 畫面顯示 ======
    cv2.putText(frame, f"{目前類別} ({信心*100:.1f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"持續: {持續秒數:.1f}s",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("我的分類器應用", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
