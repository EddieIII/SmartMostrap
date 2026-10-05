import serial
import time

arduino = serial.Serial("COM3", 115200)

time.sleep(2)

while True:
    cmd = input("1=Aedes 2=Culex 3=None  : ")
    arduino.write(cmd.encode("utf-8"))
    print("Sent:", cmd)