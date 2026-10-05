import torch
from ultralytics import YOLO

def main():
    # 1. ตรวจสอบว่าระบบมองเห็นการ์ดจอ NVIDIA (CUDA) หรือไม่
    if torch.cuda.is_available():
        print(f"🔥 กำลังใช้การ์ดจอ: {torch.cuda.get_device_name(0)}")
        device_target = 0  # เลือกใช้ GPU ตัวแรก
    else:
        print("⚠️ ไม่พบการ์ดจอ NVIDIA ระบบจะสลับไปใช้ CPU แทน")
        device_target = "cpu"

    # 2. โหลดโมเดลฐาน YOLOv8 (สามารถเปลี่ยนเป็น yolov8s.pt หรือ yolov8m.pt ได้ตามต้องการ)
    model = YOLO("yolov8n.pt")

    # 3. เริ่มต้นขั้นตอนการ Train
    model.train(
        data="data.yaml",      # ไฟล์ตั้งค่าตำแหน่ง Dataset ของคุณ
        epochs=100,            # จำนวนรอบในการ Train (ปรับเปลี่ยนได้)
        imgsz=640,             # ขนาดรูปภาพที่ใช้ Train
        device=device_target,  # บังคับให้รันบน GPU หรือ CPU ตามที่ตรวจพบ
        workers=2              # จำนวนเธรดในการโหลดข้อมูล (ปรับลดได้หากแรมคอมพิวเตอร์เต็ม)
    )

if __name__ == '__main__':
    main()
