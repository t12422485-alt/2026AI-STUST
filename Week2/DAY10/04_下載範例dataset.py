# 04 — 從 Roboflow 下載範例 dataset
#
# 用法：
#   1) 把 API_KEY 換成你自己的（怎麼拿見 03 教學）
#   2) 選你要下的 dataset（下方預設剪刀石頭布，也可以換成其他）
#   3) python 04_下載範例dataset.py
#
# 需要：pip install roboflow

import os
from pathlib import Path

# ====== 你要改的地方 ======
API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")

# 幾個經典 dataset，選一個把 `use_this` 對到那個 key
DATASETS = {
    # 剪刀石頭布（3 類，~1200 張，最快，20 分鐘 CPU 訓練可達 90%+ mAP）
    "rps": {
        "workspace": "roboflow-58fyf",
        "project":   "rock-paper-scissors-sxsw",
        "version":   14,
        "描述":     "Rock Paper Scissors 剪刀石頭布 (3 類，1237 張)",
    },
    # 海洋生物（7 類，638 張）
    "aquarium": {
        "workspace": "brad-dwyer",
        "project":   "aquarium-combined",
        "version":   2,
        "描述":     "Aquarium 水族館海洋生物 (7 類，638 張)",
    },
    # 口罩偵測（3 類：戴、沒戴、戴錯）
    "mask": {
        "workspace": "joseph-nelson",
        "project":   "mask-wearing",
        "version":   4,
        "描述":     "Mask Wearing 口罩配戴 (2 類，149 張)",
    },
    # 西洋棋（12 類 6 種棋 × 2 顏色，693 張）
    "chess": {
        "workspace": "joseph-nelson",
        "project":   "chess-pieces-new",
        "version":   23,
        "描述":     "Chess Pieces 西洋棋 (12 類，693 張)",
    },
    # ===== 人物相關 =====
    # 工地帽子偵測（2 類：helmet / person，~5000 張）— 業界最經典 PPE 教學
    "hardhat": {
        "workspace": "joseph-nelson",
        "project":   "hard-hat-workers",
        "version":   2,
        "描述":     "Hard Hat Workers 工地安全帽 (2 類，~5000 張)",
    },
    # 一般人物偵測（1 類：person）
    "person": {
        "workspace": "titulacin-tfg",
        "project":   "person-detection-9a6mk",
        "version":   1,
        "描述":     "Person Detection 純人物偵測 (1 類)",
    },
    # 跌倒偵測（適合長照 / 安養院應用）
    "fall": {
        "workspace": "fall-detection-r7ips",
        "project":   "fall-detection-3ubz1",
        "version":   1,
        "描述":     "Fall Detection 跌倒偵測 (2 類：fall / not-fall)",
    },
}

use_this = "rps"    # ← 改這裡選 dataset

# ====== 執行下載 ======
if API_KEY == "YOUR_API_KEY_HERE":
    print("=" * 60)
    print("請先設定你的 Roboflow API Key")
    print("=" * 60)
    print("方法 1) 直接改本檔上方的 API_KEY = ...")
    print("方法 2) 環境變數：")
    print('  set ROBOFLOW_API_KEY=你的key')
    print("拿 key 步驟見 03_Roboflow下載dataset教學.md")
    raise SystemExit(1)

info = DATASETS[use_this]
print(f"準備下載：{info['描述']}")

from roboflow import Roboflow

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(info["workspace"]).project(info["project"])
version = project.version(info["version"])

# 下到 datasets/<use_this>/ 目錄
BASE = Path(__file__).parent
target = BASE / "datasets" / use_this
target.parent.mkdir(exist_ok=True)

dataset = version.download("yolov11", location=str(target), overwrite=False)

print()
print(f"完成！dataset 在：{dataset.location}")
print(f"訓練時把 data 指到：{Path(dataset.location) / 'data.yaml'}")
print()
print("下一步：跑 05_訓練自己的YOLO.py")
