# 多人小遊戲 - 伺服器端
# 功能：接受多個 client 連線、廣播所有玩家狀態（位置 / 造型 / 聊天內容）
# 用法：python server.py
# 通訊：TCP，訊息為以 \n 分隔的 JSON 字串

import socket
import threading
import json
import time

HOST = "0.0.0.0"     # 監聽所有網路介面（同 WiFi 的人才連得到）
PORT = 5555
TICK_HZ = 30         # 每秒廣播 30 次
WORLD_W = 2000
WORLD_H = 2000

players = {}         # {連線編號: {"id","x","y","shape","color","chat","chat_time"}}
conns   = []         # [(socket, 連線編號)]
lock    = threading.Lock()


def 廣播迴圈():
    while True:
        time.sleep(1 / TICK_HZ)
        with lock:
            payload = {"type": "state", "players": list(players.values())}
        msg = (json.dumps(payload) + "\n").encode("utf-8")

        dead = []
        for conn, cid in list(conns):
            try:
                conn.sendall(msg)
            except OSError:
                dead.append((conn, cid))

        if dead:
            with lock:
                for conn, cid in dead:
                    if (conn, cid) in conns:
                        conns.remove((conn, cid))
                    players.pop(cid, None)
                    try:
                        conn.close()
                    except OSError:
                        pass


def 處理單一連線(conn, cid, addr):
    print(f"[{cid}] {addr} 已連線")
    conns.append((conn, cid))
    buf = b""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                t = msg.get("type")
                with lock:
                    if t == "join":
                        players[cid] = {
                            "id":    str(msg.get("id", f"P{cid}"))[:12],
                            "shape": msg.get("shape", "circle"),
                            "color": msg.get("color", [200, 50, 50]),
                            "x":     WORLD_W // 2,
                            "y":     WORLD_H // 2,
                            "chat":       "",
                            "chat_time":  0,
                        }
                        print(f"[{cid}] 加入為 {players[cid]['id']}")
                    elif t == "update" and cid in players:
                        players[cid]["x"] = max(0, min(WORLD_W, float(msg.get("x", 0))))
                        players[cid]["y"] = max(0, min(WORLD_H, float(msg.get("y", 0))))
                    elif t == "chat" and cid in players:
                        players[cid]["chat"]      = str(msg.get("text", ""))[:80]
                        players[cid]["chat_time"] = time.time()
    finally:
        with lock:
            players.pop(cid, None)
            conns[:] = [(c, i) for c, i in conns if i != cid]
        try:
            conn.close()
        except OSError:
            pass
        print(f"[{cid}] 中斷連線")


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server 開啟於 {HOST}:{PORT}（按 Ctrl+C 停止）")

    threading.Thread(target=廣播迴圈, daemon=True).start()

    cid_counter = 0
    try:
        while True:
            conn, addr = s.accept()
            cid_counter += 1
            threading.Thread(
                target=處理單一連線,
                args=(conn, cid_counter, addr),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        print("\nServer 關閉中...")
    finally:
        s.close()


if __name__ == "__main__":
    main()
