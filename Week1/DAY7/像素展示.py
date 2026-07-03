import cv2

# 滑鼠事件回呼函式
def show_pixel_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # 當滑鼠移動時
        # 複製一份原圖，避免文字疊加髒掉
        img_copy = img.copy()
        
        # OpenCV 預設是 BGR 順序
        b, g, r = img[y, x]
        
        # 準備顯示的文字
        text = f"X:{x}, Y:{y} | R:{r} G:{g} B:{b}"
        
        # 將數值畫在影像上
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Pixel Value Viewer', img_copy)

# 讀取圖片
img = cv2.imread(r'C:\Users\TSIC\Documents\project\Week1\DAY7\WIN_20260703_09_51_29_Pro.jpg'
                 ) # 請替換成你的圖片路徑

if img is None:
    print("無法讀取圖片")
    exit()

cv2.namedWindow('Pixel Value Viewer')
# 綁定滑鼠事件
cv2.setMouseCallback('Pixel Value Viewer', show_pixel_rgb)

# 初始顯示
cv2.imshow('Pixel Value Viewer', img)
cv2.waitKey(0)
cv2.destroyAllWindows()