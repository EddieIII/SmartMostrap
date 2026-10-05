from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
print("รายชื่อคลาสยุงในโมเดลของคุณคือ:")
print(model.names)
