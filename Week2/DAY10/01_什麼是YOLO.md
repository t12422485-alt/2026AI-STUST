# 01 — 什麼是 YOLO

**目標**：10 分鐘理解 YOLO 為什麼快、為什麼準、以及和 DAY9 的 MediaPipe / EfficientDet 比起來差在哪。

---

## 名字的由來

**YOLO = You Only Look Once**（你只需要看一次）

比它更早的物件偵測方法（例如 R-CNN、Fast R-CNN、Faster R-CNN）流程是：

```
圖片 → 找出「可能有東西」的候選區域 (幾百到幾千個框)
     → 對每個框跑一次 CNN 判斷是什麼
     → 慢
```

YOLO 的想法：**別分兩步，一次搞定**。

```
圖片 → 一個 CNN 直接吐出所有的框 + 類別 + 信心
     → 一次前向傳播就結束
     → 快
```

這叫 **one-stage detector**（一階段偵測），R-CNN 系列叫 **two-stage detector**。

---

## YOLO 怎麼「一次」抓出所有東西

想像把圖片切成 SxS 的格子（例如 13x13）。每個格子負責預測：

- **B 個 bounding box**（座標 x, y, w, h + 信心分數）
- 每個 box 屬於 **C 個類別** 的機率

輸出是一個 `S × S × (B × 5 + C)` 的張量。訓練時用一個超大 loss 同時懲罰：
- 位置不準
- 大小不對
- 類別分錯
- 有物件卻沒偵測到（漏掉）
- 沒物件卻偵測到（假警報）

推論時用 **NMS**（非極大值抑制）去掉重疊的重複框。

---

## 一句話比較

| 方法 | 速度 | 準度 | 適用 |
|---|---|---|---|
| Faster R-CNN | 慢 | 高 | 學術基準 |
| YOLO | **快** | 中→高 | 即時應用（自駕、監控）|
| SSD | 快 | 中 | 移動裝置 |
| DETR (transformer) | 慢 | 很高 | 研究 |

---

## v1 → 11 演進速覽

- **YOLOv1 (2015, Redmon)** 開山之作，快但小物件差
- **YOLOv2 / v3 (2016–2018, Redmon)** 加 anchor box、多尺度預測
- **YOLOv4 (2020, Bochkovskiy)** 各種訓練技巧（Mosaic 增強、CIoU loss）
- **YOLOv5 (2020, Ultralytics)** 純 PyTorch 實作，超好用
- **YOLOv6 / v7 (百度 / 清華 2022)** 
- **YOLOv8 (2023, Ultralytics)** 統一 API 支援偵測 / 分割 / 姿態 / 分類
- **YOLOv9 / v10 (2024)** 各種學術突破
- **YOLO11 (2024-09, Ultralytics)** ← 我們用這個！

> 注意命名：Ultralytics 官方 2024/09 起把新版**去掉 v**，寫成 `YOLO11` 而不是 `YOLOv11`。權重檔叫 `yolo11n.pt`。

## YOLO11 有 5 種大小

| 模型 | 大小 | 適用 |
|---|---|---|
| `yolo11n.pt` (nano) | 5.4MB | **我們用這個**，快、CPU 也能跑 |
| `yolo11s.pt` (small) | 19MB | |
| `yolo11m.pt` (medium) | 40MB | |
| `yolo11l.pt` (large) | 51MB | |
| `yolo11x.pt` (xlarge) | 114MB | 準度最高 |

**小數據集就用 nano**，訓練夠快、過擬合風險低。

---

## 和 DAY9 兩個方法的差別

| | DAY9 03 (TM 分類) | DAY9 08 (MediaPipe OD) | DAY10 YOLO11 |
|---|---|---|---|
| 任務 | 整張圖分類 | 物件偵測 | 物件偵測 |
| 框自己畫 | ✗ | ✓ 預訓練 80 類 | ✓ **自己選類別** |
| 自訓練 | ✓（用 TM 網頁 GUI）| ✗ 只能用預訓練 | ✓ **Ultralytics 一鍵訓練** |
| 資料集來源 | 網頁錄影 | 官方 COCO 資料 | **Roboflow Universe** |

**DAY10 是給你「工業用」的路子**：Roboflow 標資料 → YOLO 訓練 → 部署。

---

## 下一步

- 讀 `03_Roboflow下載dataset教學.md` 了解怎麼拿到資料
- 或直接跑 `02_預訓練yolo11n推論.py` 立刻玩一玩
