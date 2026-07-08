# 06 — 讀訓練結果，畫關鍵 metric
#
# 訓練完 Ultralytics 會把所有結果存到 runs/train/<你的 run>/：
#   results.csv           每個 epoch 的 loss / mAP / precision / recall
#   confusion_matrix.png  混淆矩陣（哪些類別會被搞混）
#   labels.jpg            訓練集標籤分布
#   val_batch*.jpg        驗證集預測 vs 真實對照
#   weights/best.pt       最佳模型（拿去推論）
#   weights/last.pt       最後一輪的模型
#
# 這支腳本讀 results.csv，畫學習曲線給你看

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).parent
# 改這裡：指向你的訓練結果資料夾（跑 05 時終端會 print 出來）
RUN_DIR = BASE / "runs" / "train"

results_csv = RUN_DIR / "results.csv"
if not results_csv.exists():
    raise FileNotFoundError(f"找不到 {results_csv}，先跑 05_訓練自己的YOLO.py")

df = pd.read_csv(results_csv)
# Ultralytics 的欄位名前面通常有空白，清掉
df.columns = [c.strip() for c in df.columns]
print("可用欄位：", df.columns.tolist())

# ====== 畫 4 張學習曲線：loss / mAP / precision / recall ======
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"訓練結果：{RUN_DIR.name}", fontsize=14, fontweight="bold")

epochs = df["epoch"]

# 1) box loss + cls loss + dfl loss
ax = axes[0, 0]
for col, name in [("train/box_loss", "訓練 box"),
                  ("train/cls_loss", "訓練 cls"),
                  ("val/box_loss",   "驗證 box"),
                  ("val/cls_loss",   "驗證 cls")]:
    if col in df.columns:
        ax.plot(epochs, df[col], label=name)
ax.set_title("Loss 曲線")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.legend(); ax.grid(alpha=0.3)

# 2) mAP
ax = axes[0, 1]
for col, name in [("metrics/mAP50(B)",    "mAP@0.5"),
                  ("metrics/mAP50-95(B)", "mAP@0.5:0.95")]:
    if col in df.columns:
        ax.plot(epochs, df[col], label=name, marker="o", markersize=3)
ax.set_title("mAP (越高越好)")
ax.set_xlabel("epoch"); ax.set_ylabel("mAP")
ax.legend(); ax.grid(alpha=0.3)

# 3) precision
ax = axes[1, 0]
if "metrics/precision(B)" in df.columns:
    ax.plot(epochs, df["metrics/precision(B)"], color="#2ecc71", marker="o", markersize=3)
ax.set_title("Precision (準確率：判為正的裡面真的正的比例)")
ax.set_xlabel("epoch"); ax.set_ylabel("precision")
ax.grid(alpha=0.3)

# 4) recall
ax = axes[1, 1]
if "metrics/recall(B)" in df.columns:
    ax.plot(epochs, df["metrics/recall(B)"], color="#e74c3c", marker="o", markersize=3)
ax.set_title("Recall (召回率：所有真的正的裡面被抓到的比例)")
ax.set_xlabel("epoch"); ax.set_ylabel("recall")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ====== 印最終指標 ======
last = df.iloc[-1]
print()
print("=" * 60)
print(f"最終 epoch {int(last['epoch'])} 指標")
print("=" * 60)
for col in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/precision(B)", "metrics/recall(B)"]:
    if col in df.columns:
        print(f"  {col:30s} {last[col]:.4f}")

print()
print("如果訓練結果不好：")
print("  - 增加 epochs（05 的 EPOCHS 調到 50~100）")
print("  - 檢查 data.yaml 的類別名是否正確")
print("  - 檢查 confusion_matrix.png 看哪兩類搞混，補資料")
print("  - 資料量太少可以做手動 augmentation 或用 Roboflow 內建的 augmentation")
print()
print(f"混淆矩陣圖：{RUN_DIR / 'confusion_matrix.png'}")
print(f"預測對照：{RUN_DIR / 'val_batch0_pred.jpg'}")
