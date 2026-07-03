import cv2

# 1. 建立 VideoCapture 物件。參數 0 通常代表內建的預設鏡頭。
# 如果有接外接鏡頭，可以嘗試改為 1, 2 等。
cap = cv2.VideoCapture(0)

# 檢查鏡頭是否成功開啟
if not cap.isOpened():
    print("錯誤：無法開啟視訊鏡頭。")
    exit()

print("鏡頭已開啟！按下 'q' 鍵可結束程式。")

while True:
    # 2. 逐影格（Frame）讀取影像
    # ret 是布林值，代表成功讀取與否；frame 是該影格的影像矩陣
    ret, frame = cap.read()

    # 如果讀取失敗，就跳出迴圈
    if not ret:
        print("錯誤：無法接收影像畫面。")
        break

    # 3. 顯示影像畫面，視窗名稱為 'Webcam'
    cv2.imshow('Webcam', frame)

    # 4. 等待 1 毫秒，並偵測是否按下了 'q' 鍵
    # 0xFF 是為了只取最後一個位元組，確保在不同平台上鍵盤碼正確
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 釋放鏡頭資源並關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()