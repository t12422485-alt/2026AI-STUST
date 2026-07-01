# 多人小遊戲 - 客戶端 (pygame)
# 功能：
#   - 加入時選 造型 / 顏色 / ID
#   - WASD 移動
#   - 螢幕左下角有小地圖
#   - 按 Enter 開聊天欄，Enter 送出，聊天內容變成頭上的泡泡
# 用法：python client.py

import socket
import threading
import json
import time
import sys

import pygame

# ====== 網路設定 ======
SERVER_HOST = input("Server IP (預設 127.0.0.1)：").strip() or "127.0.0.1"
SERVER_PORT = 5555

# ====== 角色選擇 ======
造型清單 = {"1": "circle", "2": "square", "3": "triangle"}
顏色清單 = {
    "1": [220,  50,  50],   # 紅
    "2": [ 50, 100, 220],   # 藍
    "3": [ 50, 200,  80],   # 綠
    "4": [240, 220,  60],   # 黃
    "5": [180,  80, 220],   # 紫
}
print("造型：1=圓形  2=方形  3=三角形")
shape = 造型清單.get(input("造型：").strip(), "circle")
print("顏色：1=紅  2=藍  3=綠  4=黃  5=紫")
color = 顏色清單.get(input("顏色：").strip(), [220, 50, 50])
player_id = (input("ID (名字)：").strip() or f"P{int(time.time()) % 1000}")[:12]

# ====== 常數 ======
WORLD_W, WORLD_H = 2000, 2000
SCREEN_W, SCREEN_H = 1000, 700
MINIMAP_SIZE = 180
CHAT_DURATION = 5      # 聊天泡泡秒數
SPEED = 260            # 像素/秒
PLAYER_SIZE = 16

# ====== 建立連線 ======
sock = socket.socket()
try:
    sock.connect((SERVER_HOST, SERVER_PORT))
except OSError as e:
    print(f"連線失敗：{e}")
    sys.exit(1)


def 傳送(msg):
    try:
        sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
    except OSError:
        pass


傳送({"type": "join", "id": player_id, "shape": shape, "color": color})

# ====== 共享狀態 ======
my_x, my_y = WORLD_W / 2, WORLD_H / 2
all_players = []
state_lock = threading.Lock()
running = True


def 接收迴圈():
    global all_players
    buf = b""
    while running:
        try:
            data = sock.recv(8192)
        except OSError:
            return
        if not data:
            return
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                msg = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if msg.get("type") == "state":
                with state_lock:
                    all_players = msg["players"]


threading.Thread(target=接收迴圈, daemon=True).start()

# ====== pygame 初始化 ======
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption(f"Multiplayer - {player_id}")
try:
    font = pygame.font.SysFont("Microsoft JhengHei", 16)
    big_font = pygame.font.SysFont("Microsoft JhengHei", 22)
except Exception:
    font = pygame.font.Font(None, 20)
    big_font = pygame.font.Font(None, 26)
clock = pygame.time.Clock()


def 畫玩家(surf, p, sx, sy, size=PLAYER_SIZE):
    col = tuple(p["color"])
    shape = p["shape"]
    if shape == "circle":
        pygame.draw.circle(surf, col, (sx, sy), size)
        pygame.draw.circle(surf, (0, 0, 0), (sx, sy), size, 2)
    elif shape == "square":
        rect = pygame.Rect(sx - size, sy - size, size * 2, size * 2)
        pygame.draw.rect(surf, col, rect)
        pygame.draw.rect(surf, (0, 0, 0), rect, 2)
    elif shape == "triangle":
        pts = [(sx, sy - size),
               (sx - size, sy + size),
               (sx + size, sy + size)]
        pygame.draw.polygon(surf, col, pts)
        pygame.draw.polygon(surf, (0, 0, 0), pts, 2)


def 畫聊天泡泡(surf, sx, sy, text):
    if not text:
        return
    text_surf = font.render(text, True, (0, 0, 0))
    w, h = text_surf.get_size()
    pad = 6
    bubble = pygame.Rect(sx - w // 2 - pad,
                         sy - PLAYER_SIZE - 12 - h - pad,
                         w + pad * 2, h + pad * 2)
    pygame.draw.rect(surf, (255, 255, 255), bubble, border_radius=8)
    pygame.draw.rect(surf, (0, 0, 0), bubble, 2, border_radius=8)
    # 三角尖角
    tip = [(sx - 5, bubble.bottom),
           (sx + 5, bubble.bottom),
           (sx,     bubble.bottom + 6)]
    pygame.draw.polygon(surf, (255, 255, 255), tip)
    pygame.draw.line(surf, (0, 0, 0), tip[0], tip[2], 2)
    pygame.draw.line(surf, (0, 0, 0), tip[1], tip[2], 2)
    surf.blit(text_surf, (bubble.x + pad, bubble.y + pad))


# ====== 主迴圈狀態 ======
chat_active = False
chat_text = ""

try:
    while running:
        dt = clock.tick(60) / 1000
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if chat_active:
                    if event.key == pygame.K_RETURN:
                        if chat_text.strip():
                            傳送({"type": "chat", "text": chat_text.strip()})
                        chat_text = ""
                        chat_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        chat_text = chat_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        chat_text = ""
                        chat_active = False
                    elif event.unicode and event.unicode.isprintable():
                        chat_text += event.unicode
                else:
                    if event.key == pygame.K_RETURN:
                        chat_active = True

        # 移動（只在非聊天模式下）
        if not chat_active:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]: my_y -= SPEED * dt
            if keys[pygame.K_s]: my_y += SPEED * dt
            if keys[pygame.K_a]: my_x -= SPEED * dt
            if keys[pygame.K_d]: my_x += SPEED * dt
            my_x = max(0, min(WORLD_W, my_x))
            my_y = max(0, min(WORLD_H, my_y))
            傳送({"type": "update", "x": my_x, "y": my_y})

        # 相機以自己為中心
        cam_x = my_x - SCREEN_W / 2
        cam_y = my_y - SCREEN_H / 2

        screen.fill((240, 240, 220))

        # 地圖格線
        grid = 100
        gx0 = int(-cam_x % grid)
        gy0 = int(-cam_y % grid)
        for x in range(gx0, SCREEN_W, grid):
            pygame.draw.line(screen, (210, 210, 190), (x, 0), (x, SCREEN_H))
        for y in range(gy0, SCREEN_H, grid):
            pygame.draw.line(screen, (210, 210, 190), (0, y), (SCREEN_W, y))

        # 世界邊界
        pygame.draw.rect(screen, (150, 150, 150),
                         pygame.Rect(-cam_x, -cam_y, WORLD_W, WORLD_H), 3)

        with state_lock:
            snapshot = list(all_players)

        # 畫玩家
        for p in snapshot:
            sx = int(p["x"] - cam_x)
            sy = int(p["y"] - cam_y)
            if -50 < sx < SCREEN_W + 50 and -50 < sy < SCREEN_H + 50:
                畫玩家(screen, p, sx, sy)
                # ID 名字
                id_surf = font.render(p["id"], True, (0, 0, 0))
                screen.blit(id_surf, (sx - id_surf.get_width() // 2, sy + PLAYER_SIZE + 4))
                # 聊天泡泡
                if p.get("chat") and now - p.get("chat_time", 0) < CHAT_DURATION:
                    畫聊天泡泡(screen, sx, sy, p["chat"])

        # 小地圖 (左下角)
        mm_x, mm_y = 10, SCREEN_H - MINIMAP_SIZE - 10
        mm_rect = pygame.Rect(mm_x, mm_y, MINIMAP_SIZE, MINIMAP_SIZE)
        pygame.draw.rect(screen, (30, 30, 30), mm_rect)
        pygame.draw.rect(screen, (255, 255, 255), mm_rect, 2)
        # 相機視野框
        view_x = mm_x + (cam_x) * MINIMAP_SIZE / WORLD_W
        view_y = mm_y + (cam_y) * MINIMAP_SIZE / WORLD_H
        view_w = SCREEN_W * MINIMAP_SIZE / WORLD_W
        view_h = SCREEN_H * MINIMAP_SIZE / WORLD_H
        pygame.draw.rect(screen, (255, 255, 0),
                         pygame.Rect(view_x, view_y, view_w, view_h), 1)
        # 每個玩家一個點
        for p in snapshot:
            px = mm_x + int(p["x"] * MINIMAP_SIZE / WORLD_W)
            py = mm_y + int(p["y"] * MINIMAP_SIZE / WORLD_H)
            pygame.draw.circle(screen, tuple(p["color"]), (px, py), 3)

        # 聊天輸入欄
        if chat_active:
            box = pygame.Rect(10, SCREEN_H - 40, SCREEN_W - 20, 30)
            pygame.draw.rect(screen, (255, 255, 255), box)
            pygame.draw.rect(screen, (0, 0, 0), box, 2)
            text_surf = big_font.render("> " + chat_text, True, (0, 0, 0))
            screen.blit(text_surf, (box.x + 6, box.y + 3))
        else:
            hint = font.render("按 Enter 打字聊天  |  WASD 移動", True, (80, 80, 80))
            screen.blit(hint, (10, SCREEN_H - 25))

        # 玩家數
        cnt_surf = font.render(f"線上：{len(snapshot)}", True, (0, 0, 0))
        screen.blit(cnt_surf, (SCREEN_W - cnt_surf.get_width() - 10, 10))

        pygame.display.flip()

finally:
    running = False
    pygame.quit()
    try:
        sock.close()
    except OSError:
        pass
