# 05 — 訓練你自己的 YOLO11n
#
# 這支腳本會：
#   1) 自動掃 datasets/ 底下所有 data.yaml，列給你選
#   2) 偵測有沒有 GPU，沒有就 CPU 訓練（會慢很多）
#   3) 依 GPU 顯存自動建議 batch size
#   4) 訓練結果永遠存到 DAY10/runs/ 底下（絕對路徑，跟你 cwd 無關）
#
# 訓練時間參考（YOLO11n、640x640、20 epochs、1200 張左右資料集）：
#   - CPU (i5/i7 筆電)          30-90 分鐘
#   - GTX 1660 6GB              6-10 分鐘
#   - RTX 3060 12GB             3-5 分鐘
#   - RTX 4090 24GB             1-2 分鐘
#
# 環境確認：
#   pip install -r requirements.txt

from pathlib import Path
import sys

import torch
from ultralytics import YOLO

BASE       = Path(__file__).parent
DATASETS   = BASE / "datasets"
PROJECT_DIR = BASE / "runs"        # 絕對路徑，訓練結果永遠在 DAY10/runs/


# ====== 訓練參數（可微調）======
EPOCHS       = 20         # 訓練回合。20 是入門合理值。看 mAP 曲線沒收斂再加
IMAGE_SIZE   = 640        # 影像解析度。320 快、640 準、1280 GPU 才吃得動
RUN_NAME     = "train"    # 存到 runs/train, runs/train2, ...
PATIENCE     = 10         # 連續 N epoch mAP 沒進步就早停（early stopping）


# ====== 自動找可用的 dataset ======
def 找所有dataset():
    """回傳 [(name, data_yaml_path), ...]"""
    if not DATASETS.exists():
        return []
    results = []
    for yaml in sorted(DATASETS.rglob("data.yaml")):
        # yaml 檔案的父資料夾名字當 dataset 名字
        results.append((yaml.parent.name, yaml))
    return results


def 選dataset(datasets):
    if not datasets:
        print()
        print("=" * 60)
        print("找不到任何 dataset")
        print("=" * 60)
        print(f"預期位置: {DATASETS}/<名字>/data.yaml")
        print()
        print("有兩條路取得資料集：")
        print("  A) 跑 04_下載範例dataset.py 從 Roboflow 下載")
        print("  B) 用 Label Studio 標完後，跑 convert_ls_json_to_yolo.py 轉檔")
        sys.exit(1)

    if len(datasets) == 1:
        name, path = datasets[0]
        print(f"只找到一個 dataset：{name}")
        return path

    print()
    print("找到多個 dataset，選一個訓練：")
    for i, (name, path) in enumerate(datasets, 1):
        # 順便顯示類別資訊
        try:
            import yaml as yaml_lib
            info = yaml_lib.safe_load(open(path, encoding="utf-8"))
            nc = info.get("nc", "?")
            names = info.get("names", [])
            print(f"  [{i}] {name:20s}  {nc} 類：{names}")
        except Exception:
            print(f"  [{i}] {name}")
    print()

    while True:
        try:
            choice = input(f"選 [1-{len(datasets)}]（Enter 用 1）: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(1)
        if not choice:
            return datasets[0][1]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx][1]
        except ValueError:
            pass
        print("請輸入有效數字")


# ====== 依 GPU 顯存建議 batch size ======
def 建議batch_size():
    if not torch.cuda.is_available():
        return 8, "cpu", None
    device_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # 粗略對照：640 訓練 YOLO11n 大約需要 (batch × 1.5) GB
    if total_gb >= 20:
        batch = 32
    elif total_gb >= 10:
        batch = 16
    elif total_gb >= 6:
        batch = 8
    elif total_gb >= 4:
        batch = 4
    else:
        batch = 2
    return batch, "cuda:0", f"{device_name} ({total_gb:.1f} GB)"


def main():
    print("=" * 60)
    print("YOLO11n 訓練 (Ultralytics)")
    print("=" * 60)

    # 1) 選 dataset
    datasets = 找所有dataset()
    data_yaml = 選dataset(datasets)
    print(f"使用 dataset：{data_yaml}")

    # 2) 顯示裝置與 batch size 建議
    batch, device, gpu_info = 建議batch_size()
    print()
    if gpu_info:
        print(f"偵測到 GPU：{gpu_info}")
        print(f"建議 batch size：{batch}")
        print("  若訓練中 out of memory，把 batch 調小（或關其他吃 GPU 的程式）")
    else:
        print("沒偵測到 GPU，使用 CPU 訓練")
        print(f"建議 batch size：{batch}（CPU 也不宜太大）")
        print("  預估時間：小資料集 30-60 分鐘，中資料集 1-3 小時")

    print()
    print(f"訓練參數：epochs={EPOCHS}, imgsz={IMAGE_SIZE}, patience={PATIENCE}")
    print(f"結果存到：{PROJECT_DIR}")
    print()

    # 3) 讓學員確認（避免不小心啟動長時間訓練）
    try:
        proceed = input("按 Enter 開始訓練（Ctrl+C 取消）：").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    # 4) 訓練
    model = YOLO("yolo11n.pt")   # 第一次跑會自動下載 5.5MB
    results = model.train(
        data       = str(data_yaml),
        epochs     = EPOCHS,
        imgsz      = IMAGE_SIZE,
        batch      = batch,
        device     = device,
        project    = str(PROJECT_DIR),   # 絕對路徑！不管 cwd 在哪都存這
        name       = RUN_NAME,
        exist_ok   = False,              # 每次跑新 folder：train, train2, ...
        patience   = PATIENCE,
        verbose    = True,
    )

    # 5) 結果摘要
    save_dir = Path(results.save_dir)
    print()
    print("=" * 60)
    print("訓練完成")
    print("=" * 60)
    print(f"最佳權重：{save_dir / 'weights' / 'best.pt'}")
    print(f"最後權重：{save_dir / 'weights' / 'last.pt'}")
    print(f"訓練圖表：{save_dir / 'results.png'}")
    print(f"混淆矩陣：{save_dir / 'confusion_matrix.png'}")
    print()
    print("下一步：")
    print("  06_評估訓練結果.py     看 mAP / Loss 曲線")
    print("  07_自訓練模型推論.py   接 webcam 有 GUI 版")
    print("  08_webcam串流推論.py   接 webcam 串流到瀏覽器（推薦）")


if __name__ == "__main__":
    main()
