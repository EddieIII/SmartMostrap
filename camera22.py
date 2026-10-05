from ultralytics import YOLO
import cv2
import serial
import time

# ======================
# SETTINGS (ปรับได้ตรงนี้)
# ======================
MODEL_PATH = "C:\\Users\\zabih\\Downloads\\Mosquito_New-20261004T042041Z-1-001\\Mosquito_New\\runs\\detect\\train-5\\weights\\best.pt"
COM_PORT = "COM3"
BAUD = 115200

CONF_THRESHOLD = 0.30     # ความมั่นใจขั้นต่ำของ YOLO
IOU_THRESHOLD = 0.45
DONE_TIMEOUT = 10         # รอ "DONE" จาก Arduino นานสุดกี่วินาที (กันค้างถ้าไม่มีตอบกลับ)
SETTLE_SECONDS = 2        # หลัง servo เสร็จ รออีกกี่วินาทีให้ยุงตัวใหม่เข้ามา/ภาพนิ่ง
CONFIRM_FRAMES = 3        # ต้องเจอคลาสเดิมติดกันกี่เฟรมก่อนส่ง (กันตรวจจับผิดพลาด)

# label ของโมเดล -> คำสั่งที่ส่งไป Arduino
# (รองรับทั้ง "ades" ตามโมเดลเดิม และ "aedes")
COMMANDS = {
    "ades": b"1",
    "aedes": b"1",
    "culex": b"2",
}
UNKNOWN_COMMAND = b"3"    # ยุงที่ไม่ใช่ Aedes/Culex (Unknown)

COLORS = {
    b"1": (0, 0, 255),     # Aedes  = แดง
    b"2": (255, 255, 0),   # Culex  = ฟ้าอ่อน
    b"3": (0, 255, 0),     # Unknown = เขียว
}
NAMES = {b"1": "AEDES", b"2": "CULEX", b"3": "UNKNOWN"}


def label_to_command(label: str) -> bytes:
    return COMMANDS.get(label.lower(), UNKNOWN_COMMAND)


# ======================
# YOLO MODEL
# ======================
model = YOLO(MODEL_PATH)

# ======================
# ARDUINO
# ======================
try:
    arduino = serial.Serial(COM_PORT, BAUD, timeout=0.1)
    time.sleep(2)  # รอ Arduino รีเซ็ตหลังเปิดพอร์ต
    print("✅ Arduino Connected")
except Exception as e:
    print("❌ Arduino Error")
    print(e)
    raise SystemExit(1)

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลดภาพค้าง/ดีเลย์

if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    arduino.close()
    raise SystemExit(1)

print("✅ Mosquito Sorter Started (กด q เพื่อออก)")

# ======================
# STATE
# ======================
pause_until = 0.0          # เวลาที่จะกลับมาตรวจจับได้
waiting_done = False       # กำลังรอ Arduino ตอบ DONE
sent_at = 0.0
last_command = None
streak_command = None      # คำสั่งที่เจอติดกันอยู่
streak_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()

        # อ่านข้อความที่ Arduino ตอบกลับ (ถ้ามี) เพื่อ debug
        got_done = False
        while arduino.in_waiting:
            msg = arduino.readline().decode(errors="ignore").strip()
            if msg:
                print(f"[Arduino] {msg}")
                if msg == "DONE":
                    got_done = True

        # ----------------------
        # ช่วงพัก: ไม่รัน YOLO แต่ยังแสดงภาพสด
        # ----------------------
        if waiting_done:
            if got_done:
                waiting_done = False
                pause_until = now + SETTLE_SECONDS
            elif now - sent_at > DONE_TIMEOUT:
                print("⚠ ไม่ได้รับ DONE จาก Arduino (timeout) - กลับมาตรวจจับต่อ")
                waiting_done = False
                pause_until = now + SETTLE_SECONDS

        if waiting_done or now < pause_until:
            status = NAMES.get(last_command, "")
            text = (
                f"SENT {status} - SERVO WORKING..."
                if waiting_done
                else f"SENT {status} - WAIT {pause_until - now:.1f}s"
            )
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 200, 255),
                2,
            )
            cv2.imshow("Mosquito Sorter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ----------------------
        # ตรวจจับ
        # ----------------------
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        best = None  # (conf, command, label)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = model.names[int(box.cls[0])]
                command = label_to_command(label)
                color = COLORS[command]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"{NAMES[command]} {conf * 100:.1f}%",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                # เลือกตัวที่มั่นใจที่สุดในเฟรมนี้
                if best is None or conf > best[0]:
                    best = (conf, command, label)

        # ----------------------
        # ยืนยันผลติดกันหลายเฟรมแล้วค่อยส่ง
        # ----------------------
        if best is None:
            # ไม่มียุงในเฟรม -> ไม่ส่งอะไร รีเซ็ตตัวนับ
            streak_command = None
            streak_count = 0
        else:
            _, command, label = best
            if command == streak_command:
                streak_count += 1
            else:
                streak_command = command
                streak_count = 1

            if streak_count >= CONFIRM_FRAMES:
                arduino.reset_input_buffer()  # ทิ้งข้อความเก่าที่ค้าง
                arduino.write(command)
                arduino.flush()
                last_command = command
                waiting_done = True
                sent_at = time.time()
                print(
                    f"Detected: {label} ({best[0] * 100:.1f}%) "
                    f"➡ SEND {command.decode()} | หยุดตรวจจับจนกว่า servo เสร็จ"
                )
                streak_command = None
                streak_count = 0

        cv2.imshow("Mosquito Sorter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    arduino.close()
    cv2.destroyAllWindows()