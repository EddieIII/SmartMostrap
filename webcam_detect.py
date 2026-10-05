import cv2
from ultralytics import YOLO

# 1. โหลดโมเดล (ตรวจสอบให้แน่ใจว่าบรรทัดนี้ไม่ได้ถูกลบหรือใส่เครื่องหมาย # คอมเมนต์ไว้)
# จากเดิม: model = YOLO('yolov8n.pt')
# ให้แก้ไขเป็นพาธไฟล์โมเดลยุงที่คุณเทรนสำเร็จ:
model = YOLO('runs/detect/train/weights/best.pt')



# 1. บังคับเรียกกล้องนอก (เลข 2) ผ่านระบบ DirectShow เพื่อเคลียร์ภาพค้าง
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

# 2. ตั้งค่า MJPG ป้องกันภาพซ่าลายเส้นเหมือนเดิม
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)




if not cap.isOpened():
    print("ไม่สามารถเปิดกล้อง Webcam เบอร์ 3 ได้ (หากไม่ขึ้น ให้ลองเปลี่ยนเลขเป็น   2)")
    exit()      

print("กำลังเปิดกล้อง... (หากต้องการปิดหน้าต่างกล้อง ให้กดปุ่ม 'q' ที่คีย์บอร์ด)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถดึงภาพจากกล้องได้")
        break

    # 4. ส่งเฟรมภาพให้โมเดลประมวลผล (มีตัวแปร model มารับรองแล้ว)
    results = model(frame, conf=0.05, stream=True)

    # วาดกล่องคำทำนายลงบนหน้าจอวิดีโอสด
    for r in results:
        annotated_frame = r.plot()

    cv2.imshow("Mosquito Detection (Webcam)", annotated_frame)

    # กดปุ่ม 'q' บนคีย์บอร์ดเพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
