# Label Studio 一鍵 export 含圖 wrapper server
#
# 為什麼要這個：
#   LS 官方的 YOLO export 不含原始圖檔（只給標籤 + URL）
#   學員從區網 export 下來要另外抓圖，很麻煩
#   這支 server 直接讀 LS 的 SQLite + media folder，打包成完整的 YOLO 訓練 zip
#
# 使用方式：
#   老師（同一台跑 LS 的機器）：
#     python ls_export_wrapper.py
#   學員（任何瀏覽器）：
#     打開 http://192.168.1.102:8082/
#     看到專案清單 → 點下載 → 拿到含圖的 zip
#     解壓縮到 Week2/DAY10/datasets/<你的名字>/ 就能訓練
#
# 需要：pip install flask

import io
import json
import os
import random
import sqlite3
import zipfile
from pathlib import Path

from flask import Flask, render_template_string, send_file, Response

# ====== 設定（若你的 LS 資料夾位置不同，改這裡）======
LS_DB       = Path(os.getenv("LS_DB", r"C:\Users\TSIC\AppData\Local\label-studio\label-studio\label_studio.sqlite3"))
LS_MEDIA    = Path(os.getenv("LS_MEDIA", r"C:\Users\TSIC\AppData\Local\label-studio\label-studio\media"))
PORT        = int(os.getenv("PORT", 8082))

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.15

app = Flask(__name__)


# ====== 讀 LS SQLite 拿資料 ======
def 列出所有專案():
    con = sqlite3.connect(LS_DB)
    cur = con.execute("SELECT id, title FROM project ORDER BY id")
    projects = []
    for pid, title in cur.fetchall():
        cur2 = con.execute("SELECT COUNT(*) FROM task WHERE project_id=?", (pid,))
        total = cur2.fetchone()[0]
        cur3 = con.execute("""
            SELECT COUNT(DISTINCT t.id) FROM task t
            JOIN task_completion c ON c.task_id = t.id
            WHERE t.project_id=? AND c.was_cancelled=0
        """, (pid,))
        labeled = cur3.fetchone()[0]
        projects.append({"id": pid, "title": title, "total": total, "labeled": labeled})
    con.close()
    return projects


def 讀某專案的task和標註(project_id):
    """回傳 [(task_dict, [annotation_result_dict, ...]), ...]"""
    con = sqlite3.connect(LS_DB)
    cur = con.execute("SELECT id, data FROM task WHERE project_id=?", (project_id,))
    tasks = {tid: json.loads(data) for tid, data in cur.fetchall()}

    cur = con.execute("""
        SELECT task_id, result FROM task_completion
        WHERE task_id IN (SELECT id FROM task WHERE project_id=?)
          AND was_cancelled=0
    """, (project_id,))
    task_annos = {}
    for tid, result_json in cur.fetchall():
        results = json.loads(result_json) if isinstance(result_json, str) else result_json
        task_annos.setdefault(tid, []).extend(results if isinstance(results, list) else [results])

    con.close()

    out = []
    for tid, data in tasks.items():
        annos = task_annos.get(tid, [])
        if not annos:
            continue
        out.append((data, annos))
    return out


# ====== 轉 YOLO 格式（跟 convert_ls_json_to_yolo.py 同邏輯）======
def 轉bbox_pct為yolo(x, y, w, h):
    return (x + w / 2) / 100, (y + h / 2) / 100, w / 100, h / 100


def 找圖檔實體路徑(image_url):
    rel = image_url.split("/data/", 1)[-1]
    return LS_MEDIA / rel


def 建立zip(project_id, project_title):
    """回傳 BytesIO，內含完整 YOLO 訓練資料夾（train/valid/test 拆好 + data.yaml + 圖）"""
    task_list = 讀某專案的task和標註(project_id)
    if not task_list:
        raise ValueError(f"專案 {project_id} 沒有已標註的 task")

    # 收類別
    classes = []
    for _data, annos in task_list:
        for r in annos:
            for lbl in r.get("value", {}).get("rectanglelabels", []):
                if lbl not in classes:
                    classes.append(lbl)
    class_id = {name: i for i, name in enumerate(classes)}

    # 過濾出圖檔存在的 task
    valid = []
    for data, annos in task_list:
        # 找 image URL
        image_url = data.get("image") or next(iter(data.values()), None)
        if not image_url:
            continue
        img_path = 找圖檔實體路徑(image_url)
        if not img_path.exists():
            continue
        valid.append((data, annos, img_path))

    if not valid:
        raise ValueError(f"專案 {project_id} 找不到任何有圖的 task")

    # 隨機拆
    random.seed(42)
    random.shuffle(valid)
    n = len(valid)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits = {
        "train": valid[:n_train],
        "valid": valid[n_train:n_train + n_val],
        "test":  valid[n_train + n_val:],
    }

    # 打包成 zip（放在 memory 裡）
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for split_name, items in splits.items():
            for data, annos, img_path in items:
                # 圖
                arcname_img = f"{split_name}/images/{img_path.name}"
                z.write(img_path, arcname_img)

                # label
                lines = []
                for a in annos:
                    if a.get("type") != "rectanglelabels":
                        continue
                    v = a["value"]
                    labels = v.get("rectanglelabels", [])
                    if not labels:
                        continue
                    cls = labels[0]
                    if cls not in class_id:
                        continue
                    cx, cy, ww, hh = 轉bbox_pct為yolo(v["x"], v["y"], v["width"], v["height"])
                    lines.append(f"{class_id[cls]} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

                arcname_label = f"{split_name}/labels/{img_path.stem}.txt"
                z.writestr(arcname_label, "\n".join(lines))

        # data.yaml（用相對路徑 path: .，YOLO 訓練時自動抓當前 zip 根目錄）
        yaml = (
            f"# 由 LS wrapper server 自動產出\n"
            f"# 專案：{project_title}\n"
            f"# 類別：{classes}\n\n"
            f"path: .\n"
            f"train: train/images\n"
            f"val: valid/images\n"
            f"test: test/images\n\n"
            f"nc: {len(classes)}\n"
            f"names: {classes}\n"
        )
        z.writestr("data.yaml", yaml)

    buf.seek(0)
    return buf, len(valid), len(classes), classes


# ====== HTTP endpoints ======
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>Label Studio 一鍵 export</title>
<style>
  body { font-family: 'Microsoft JhengHei', sans-serif; max-width: 800px;
         margin: 40px auto; padding: 20px; background: #f5f5f5; }
  h1 { color: #333; }
  table { width: 100%; border-collapse: collapse; background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
  th { background: #2a5cb5; color: white; }
  a.btn { background: #34aa5b; color: white; padding: 8px 16px;
          border-radius: 4px; text-decoration: none; font-weight: bold; }
  a.btn:hover { background: #40c46b; }
  .hint { color: #777; margin: 20px 0; }
</style>
</head>
<body>
<h1>Label Studio 一鍵 export（含圖）</h1>
<div class="hint">
  點 <strong>下載 zip</strong> 拿到完整 YOLO 訓練資料集（圖 + 標籤 + data.yaml + 拆好 train/val/test）<br>
  解壓縮到 <code>Week2/DAY10/datasets/&lt;你想要的名字&gt;/</code> 就能跑 <code>05_訓練自己的YOLO.py</code>
</div>
<table>
  <tr><th>#</th><th>專案名稱</th><th>已標 / 總數</th><th>下載</th></tr>
  {% for p in projects %}
  <tr>
    <td>{{ p.id }}</td>
    <td>{{ p.title }}</td>
    <td>{{ p.labeled }} / {{ p.total }}</td>
    <td><a class="btn" href="/export/{{ p.id }}">下載 zip</a></td>
  </tr>
  {% endfor %}
</table>
</body>
</html>"""


@app.route("/")
def index():
    projects = 列出所有專案()
    return render_template_string(INDEX_HTML, projects=projects)


@app.route("/export/<int:project_id>")
def export(project_id):
    con = sqlite3.connect(LS_DB)
    row = con.execute("SELECT title FROM project WHERE id=?", (project_id,)).fetchone()
    con.close()
    if not row:
        return f"專案 {project_id} 不存在", 404
    title = row[0]

    try:
        buf, n, nc, classes = 建立zip(project_id, title)
    except ValueError as e:
        return str(e), 400

    print(f"[export] project {project_id} '{title}': {n} 張圖, {nc} 類別 {classes}")

    safe_title = "".join(c for c in title if c.isalnum() or c in "_-")[:30]
    filename = f"project-{project_id}-{safe_title}.zip"
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name=filename)


if __name__ == "__main__":
    print("=" * 60)
    print("LS Export Wrapper Server")
    print("=" * 60)
    print(f"LS SQLite:  {LS_DB}")
    print(f"LS media:   {LS_MEDIA}")
    print(f"Port:       {PORT}")
    print()
    print(f"本機打:      http://localhost:{PORT}/")
    print(f"學員同 WiFi: http://192.168.1.102:{PORT}/")
    print("=" * 60)

    if not LS_DB.exists():
        print(f"警告：找不到 LS SQLite 檔案 {LS_DB}")
        print("若你的 LS 資料夾在別的位置，用環境變數 LS_DB=... 指過去")

    app.run(host="0.0.0.0", port=PORT, threaded=True)
