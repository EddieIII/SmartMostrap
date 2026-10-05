from ultralytics import YOLO
import cv2

# ==========================================
# โหลดโมเดลที่เทรนเสร็จแล้ว
# ==========================================
model = YOLO("runs/detect/train-5/weights/best.pt")

# ==========================================
# เปิดกล้อง
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    exit()

print("✅ Mosquito Detector Started")
print("✅ กด Q เพื่อออก")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ตรวจจับ
    results = model.predict(
        source=frame,
        conf=0.30,
        verbose=False
    )

    # วาดผลลัพธ์
    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            conf = float(box.conf[0]) * 100

            cls = int(box.cls[0])

            label = model.names[cls]

            # เลือกสีตามชนิดยุง
            if label.lower() == "culex":
                color = (255, 255, 0)  # ฟ้า
            elif label.lower() == "aedes":
                color = (0, 0, 255)    # แดง
            else:
                color = (0, 255, 0)    # เขียว

            # ตีกรอบ
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # ข้อความบนกรอบ
            text = f"{label.upper()} {conf:.1f}%"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            print(
                f"พบยุง : {label.upper()} ({conf:.1f}%)"
            )

    cv2.imshow(
        "Mosquito Detector",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()