# DAY10 — YOLO11n 物件偵測 (從預訓練到自己訓練)

DAY9 教了 MediaPipe 的預訓練 OD 模型（只能認 COCO 80 類）。DAY10 進階：**訓練自己想抓的東西**。

用 Ultralytics YOLO11n（2024 年 Ultralytics 官方最新，小到 5.4MB 但準確度不俗）。

---

## 從 DAY9 → DAY10 的進化

| DAY9 (MediaPipe OD) | DAY10 (YOLO11n) |
|---|---|
| 預訓練，80 類固定 | 可自己訓練，類別自由 |
| .tflite 官方模型 | .pt PyTorch 模型 + Ultralytics 一鍵訓練 |
| 移動端友善 | 開發速度快、Roboflow 生態圈 |
| 不能 fine-tune | 幾行 `model.train()` 就能 fine-tune |

---

## 檔案

**觀念**

| 檔案 | 內容 |
|---|---|
| `01_什麼是YOLO.md` | YOLO 一鏡到底的思想、和 R-CNN 系列的差別、v1 → 11 演進速覽 |
| `03_Roboflow下載dataset教學.md` | Roboflow Universe 找資料集、開帳號、拿 API key、下載步驟 |
| `08_作業_自選dataset.md` | 期末作業指引 |

**動手**

| 檔案 | 內容 | 是否需要 GPU |
|---|---|---|
| `02_預訓練yolo11n推論.py` | 用 COCO 預訓練 `yolo11n.pt` 做即時 webcam 偵測 | ❌ CPU OK |
| `04_下載範例dataset.py` | 用 roboflow 套件下載 Rock-Paper-Scissors（3 類、~1200 張）| ❌ |
| `05_訓練自己的YOLO.py` | `model.train()` 訓練，含 CPU / GPU 自動切換 | ⚠️ CPU 慢但可以（小資料集 20 分鐘）|
| `06_評估訓練結果.py` | 訓練完看 mAP、confusion matrix、學習曲線 | ❌ |
| `07_自訓練模型推論.py` | 拿 `runs/detect/train/weights/best.pt` 上鏡頭 | ❌ |

**其他**

| 檔案 | 內容 |
|---|---|
| `requirements.txt` | 套件列表 |
| `datasets/` | 資料集放這裡（`.gitignore` 已忽略內容）|

---

## 建議順序

1. `01`（觀念）→ `02`（先玩一下預訓練，5 分鐘看到成果）
2. `03`（Roboflow 開帳號拿 API key）→ `04`（下 dataset）
3. `05`（訓練，CPU 版本先跑 20 epoch 看能不能過）
4. `06`（看訓練結果）→ `07` 或 `08`（推論驗證）
5. `08`（回頭挑一個自己的題目 e.g. 找垃圾分類、口罩偵測、料件計數）

## 訓練環境差異怎麼辦

不同學員硬體 / 環境 / 路徑都不一樣，請看 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**：
- CPU / GTX / RTX / Colab 四種情境的參數建議
- 用 Label Studio 自標的資料怎麼接進來
- 各種錯誤訊息對照表（OOM、cv2.imshow 掛掉、找不到 best.pt...）

---

## 環境準備

```powershell
conda activate STUSTPython
cd Week2\DAY10
pip install -r requirements.txt
```

第一次執行 02 會自動下載 `yolo11n.pt`（~5.5MB）。第一次訓練會自動下載 COCO 預訓練權重。
