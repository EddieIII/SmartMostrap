import cv2

print("กำลังค้นหากล้องที่เชื่อมต่ออยู่...")
for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"-> เจอตัวเลือกกล้องที่ Index {index} | ความละเอียดเริ่มต้น: {width}x{height}")
        cap.release()
