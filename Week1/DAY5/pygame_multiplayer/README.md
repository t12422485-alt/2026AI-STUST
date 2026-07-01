# pygame 多人小遊戲（Server + Client）

一個最小可玩的多人 2D 移動 + 聊天室 demo。

## 功能
- Server 一台，多個 Client 同時連線
- Client 加入時選 **造型（圓/方/三角）**、**顏色**、**ID**
- **WASD** 移動；相機以自己為中心
- 左下角有 **小地圖**（顯示所有玩家 + 目前視野框）
- 按 **Enter** 開聊天欄，再按 Enter 送出，內容變成頭頂的 **對話泡泡**（5 秒後消失）

## 安裝
```bash
pip install pygame
```

## 執行

### 1) 開 Server（一台電腦就好）
```bash
python server.py
```
會顯示 `Server 開啟於 0.0.0.0:5555`。

### 2) 開 Client（多台都可）
```bash
python client.py
```
啟動後在 CLI 依序輸入：
- Server IP（本機測試就直接 Enter 用 `127.0.0.1`；跨電腦就填 server 的區網 IP）
- 造型（1/2/3）
- 顏色（1/2/3/4/5）
- ID（名字）

跨電腦連線需求：
- Server 主機防火牆放行 TCP 5555 埠
- 兩台在同一個網段（能互相 ping）

## 通訊協定

TCP，訊息用 `\n` 分隔的 UTF-8 JSON。

| 方向 | type | 欄位 |
|---|---|---|
| Client → Server | `join`   | `id`, `shape`, `color` |
| Client → Server | `update` | `x`, `y` |
| Client → Server | `chat`   | `text` |
| Server → Client | `state`  | `players: [{id,x,y,shape,color,chat,chat_time}, ...]` |

Server 以 30Hz 廣播全體玩家狀態。座標權威由 client 自報（demo 用途，不做防作弊）。
