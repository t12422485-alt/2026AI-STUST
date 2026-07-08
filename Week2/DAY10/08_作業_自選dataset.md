# 08 — 作業：找一個屬於你的問題，訓練 YOLO 解決它

DAY10 前面帶你走完了整套流程（下 dataset → 訓練 → 評估 → 推論）。作業時把它套用到**你自己在乎的問題**上。

---

## 作業內容

**Step 1）挑一個問題**（不要挑 Roboflow 首頁那幾個範例）

想想生活裡有哪個「認出畫面裡有哪些東西 + 在哪」的場景值得自動化。例如：

- 🍎 **農業**：果實成熟度分級（未熟 / 半熟 / 全熟）
- 🚗 **交通**：機車違停偵測（機車 / 汽車 / 停車格）
- 🏥 **醫療**：X 光片異常標記
- 🏫 **校園**：宿舍寢室垃圾分類（可回收 / 廚餘 / 一般）
- ♻️ **工業**：料件計數 + 缺陷偵測（用 DAY7 的鏡頭）
- 🐕 **寵物**：寵物異常行為偵測
- 🍔 **餐廳**：餐盤上有沒有正確餐具

**Step 2）拿到 dataset**（挑一個）

- A) 到 Roboflow Universe 找一個現成的 → 04 那套下載
- B) 用 Roboflow **自己標**（免費方案 1000 張圖夠了）：
  1. 拍照或收圖（至少每類 100 張）
  2. 上傳到 Roboflow 建立 project
  3. 在網頁 label（拖框拉標籤，比 LabelImg 舒服）
  4. 發布 → export YOLOv11 → 用 API key 下載

**Step 3）訓練 + 評估 + 演示**

- 訓練後 mAP@0.5 至少 60%
- 交出：
  1. `data.yaml` 內容截圖（顯示 nc 和 names）
  2. 訓練後的 `results.png` 和 `confusion_matrix.png`
  3. 30 秒 demo 影片（用 `07_自訓練模型推論.py` 錄）
  4. 一頁 A4 說明你解決了什麼問題、資料集怎麼來、遇到的困難

---

## 加分項

**難度 ★**（做 1 個就有加分）
- 資料量 > 500 張，且是自己標的
- 加入 **NMS 後處理**：手動 IoU + 信心度融合，過濾重複框
- 用 `06` 之外自定義的 metric（例如「每張圖平均偵測到幾個」）

**難度 ★★**
- 整合進 GUI（DAY4 tkinter + DAY7 影像處理工具的架構）
- 整合進網頁（DAY6 aiohttp + WebSocket + Canvas，即時 stream 推論結果）

**難度 ★★★**
- 部署到手機（用 Ultralytics 內建 `model.export(format="tflite")` 匯出，包成 Android app 或 iOS 用）
- 用 **`sahi`**（Slice Aided Hyper Inference）處理超高解析度圖片，小物件也能抓
- 用 **深度學習之外**再加規則（例如 DAY7 形狀 + YOLO 綜合判斷）

---

## 常見坑

1. **類別不平衡**：某類 1000 張、某類 50 張 → 模型偏向多的那類。用 Roboflow 的 balance 或手動補資料
2. **標籤沒對到框**：畫框漏掉物體、或畫錯位置 → 訓練 mAP 上不去，去 `runs/train/val_batch0_pred.jpg` 對照就知道
3. **過擬合**：訓練 mAP 很好、驗證很差 → 資料不夠 / augment 打開 / epoch 別太多
4. **CPU 訓練太慢**：用 Google Colab 免費 T4 GPU（一次 12 小時），或去實驗室借 GPU

---

## 上台報告 5 分鐘該講什麼

1. 你想解決什麼問題？為什麼有意義？（1 分鐘）
2. 資料集怎麼來？多少張圖？多少類？（1 分鐘）
3. 訓練結果（mAP、confusion matrix、可視化幾張）（2 分鐘）
4. Demo 影片 + 失敗案例分析（1 分鐘）

---

**期末總複習：這一路你會的技能**

```
DAY2   Python 語法
DAY3   資料視覺化
DAY4   tkinter GUI
DAY5   socket + pygame 多人網路遊戲
DAY6   aiohttp / PostgreSQL 網頁應用
DAY7   OpenCV 電腦視覺基礎
DAY8   期中專題整合
DAY9   CNN、Teachable Machine、MediaPipe OD
DAY10  YOLO11n 自訓練物件偵測    ← 你在這裡
```

從「寫個 for 迴圈」到「訓練 AI 認你自己想抓的東西」— 恭喜完賽 🎉
