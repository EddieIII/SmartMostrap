from ultralytics import YOLO
import cv2
import serial
import time

# ======================
# YOLO MODEL
# ======================

model = YOLO(
    "runs/detect/train-5/weights/best.pt"
)

# ======================
# ARDUINO
# ======================

try:

    arduino = serial.Serial(
        "COM9",
        115200,
        timeout=1
    )

    time.sleep(2)

    print("✅ Arduino Connected")

except Exception as e:

    print("❌ Arduino Error")
    print(e)

    exit()

# ======================
# CAMERA
# ======================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("❌ เปิดกล้องไม่ได้")

    exit()

print("✅ Mosquito Sorter Started")

last_command = ""
last_trigger = 0

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    results = model.predict(
        source=frame,
        conf=0.30,
        iou=0.45,
        verbose=False
    )

    found = False

    for r in results:

        for box in r.boxes:

            found = True

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            conf = float(
                box.conf[0]
            ) * 100

            cls = int(
                box.cls[0]
            )

            label = model.names[cls].lower()

            print(
                f"Detected: {label} {conf:.1f}%"
            )

            # ส่งคำสั่งทุก 2 วิ

            if time.time() - last_trigger > 2:

                if label == "ades":

                    print("➡ SEND 1")

                    arduino.write(b'1')

                    last_command = "1"

                elif label == "culex":

                    print("➡ SEND 2")

                    arduino.write(b'2')

                    last_command = "2"

                else:

                    print("➡ SEND 3")

                    #arduino.write(b'3')

                    last_command = "3"

                last_trigger = time.time()

            # สีกรอบ

            if label == "ades":
                color = (0, 0, 255)

            elif label == "culex":
                color = (255, 255, 0)

            else:
                color = (0, 255, 0)

            # วาดกรอบ

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                frame,
                f"{label.upper()} {conf:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    if not found:

        if time.time() - last_trigger > 2:

            print("➡ SEND 3 (NO DETECTION)")

            arduino.write(b'3')

            last_trigger = time.time()

    cv2.imshow(
        "Mosquito Sorter",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()