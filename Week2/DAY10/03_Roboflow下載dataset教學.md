# 03 — Roboflow 下載 dataset 手把手

**Roboflow** 是全世界最大的物件偵測資料集共享平台。上面有幾十萬個公開資料集（`Roboflow Universe`），大部分都能直接下載成 YOLO 訓練需要的格式。

網址：<https://universe.roboflow.com/>

---

## Step 1) 註冊 Roboflow 帳號

去 <https://roboflow.com/> 點右上 `Sign Up`。免費方案就夠學生用。

---

## Step 2) 拿 API Key

登入後：
1. 點右上頭像 → `Settings`
2. 左邊選單找 `API Keys`
3. 複製 `Private API Key`（不是 Publishable Key！）
4. **把它記在便條上、或存到 .env**，不要直接寫在 push 到 GitHub 的程式碼裡

---

## Step 3) 到 Universe 找 dataset

<https://universe.roboflow.com/>

搜尋喜歡的題目：
- `rock paper scissors` 剪刀石頭布（3 類）— 教學經典
- `aquarium` 海洋生物（7 類：魚、水母、鯊魚...）
- `chess pieces` 西洋棋（12 類）
- `face mask` 口罩偵測（3 類：戴、沒戴、戴錯）
- `traffic signs` 交通號誌
- `weeds` 雜草分類（農業應用）
- `pill detection` 藥丸辨識

**挑選建議：**
- 剛開始練習 → **類別 < 10 類、樣本 < 3000 張**（CPU 訓練 20-30 分鐘可以搞定）
- 樣本數看資訊頁的 `Images`（訓練 + 驗證 + 測試 加總）

---

## 找人物 dataset 的關鍵字整理

| 用途 | 搜尋詞 |
|---|---|
| 一般人物 / 行人 | `person detection` / `pedestrian` |
| 工地安全帽 (PPE) | `hard hat` / `PPE` / `safety vest` |
| 臉部偵測 | `face detection` |
| 口罩偵測 | `face mask` / `mask wearing` |
| 跌倒偵測 (長照) | `fall detection` |
| 姿態 / 動作 | `human pose` / `action recognition` |
| 手勢 | `hand gesture` / `sign language` |
| 情緒表情 | `emotion detection` / `facial expression` |
| 打架 / 危險行為 | `fight detection` / `violence` |
| 抽菸偵測 | `smoking detection` |

**注意**：因為 Roboflow 版本會更新（v1 → v2 → v14 ...），下載前建議先到 Universe 頁面確認最新版號，改 `04_下載範例dataset.py` 的 `version` 欄位。

---

## Step 4) 下載

有兩個方式：

### 方式 A：Python 一鍵下（我們用這個）

進到某個 dataset 頁面（例如 <https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw>）
1. 點右上 `Download Dataset`
2. Format 選 **`YOLOv11`**（或 YOLOv8 也行，格式一樣）
3. 選 `Show download code`
4. 複製那段 Python 代碼，看起來像：

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(14)
dataset = version.download("yolov11")
```

把 `YOUR_KEY` 換成你自己的 API key，執行就會下載 + 解壓縮 + 建 `data.yaml`。

`04_下載範例dataset.py` 已經把上面這段包好了，你只要填 API key 就能跑。

### 方式 B：手動下載 zip

點 `Download Dataset` → 選 `YOLOv11` → 選 `Download zip to computer`。
解壓縮到 `Week2/DAY10/datasets/` 底下，資料夾裡會有：

```
你的資料集/
├── data.yaml        ← 訓練時要指定這個
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## Step 5) 看看 data.yaml 長怎樣

打開任何一個下載好的 `data.yaml`：

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3   # number of classes
names: ['Paper', 'Rock', 'Scissors']

roboflow:
  workspace: roboflow-58fyf
  ...
```

- `nc` = 類別數量
- `names` = 類別名稱列表（順序決定 class_id：0=Paper、1=Rock、2=Scissors）
- 訓練 (`05_訓練自己的YOLO.py`) 只要指到這個 `data.yaml`，Ultralytics 就會知道所有資訊

---

## 常見陷阱

- **format 一定要選 YOLOv8 或 YOLOv11**：其他格式（COCO JSON、Pascal VOC）Ultralytics 不能直接用
- **不要用 Publishable Key**：那個是給前端 demo 的，下載會失敗
- **免費方案有配額**：一個月幾百次下載請求，不過學生規模基本用不完
- **有些 dataset 是「私人」的**：找標籤 `Public` 的
- **看清楚 License**：Roboflow 上大多是 Public Domain 或 CC BY，商業用要注意
