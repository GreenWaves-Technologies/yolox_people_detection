import sys, serial, os, io
from PIL import Image
import cv2
import numpy as np


#INPUT_W=640
#INPUT_H=480

INPUT_W=320
INPUT_H=240
PIXEL_SIZE=3

UART_START_JPEG=b'\xAB\xBA'

DEBAYER=False

def main():
    #Init the window otherwise the protocol get screwed
    open_cv_image = np.zeros(shape=(INPUT_H, INPUT_W))
    cv2.imshow('Image from GAP',open_cv_image)
    cv2.waitKey(100)  
    ser = serial.Serial(
        port='/dev/ttyUSB1',\
        baudrate=3000000,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
            timeout=0.01)
    print("connected to: " + ser.portstr)
    count=0

    while True:
        read_bytes = ser.read(2)
        if read_bytes == UART_START_JPEG:
            size_payload = ser.read(1)
            while len(size_payload) < 4:
                size_payload += ser.read(1)
            data_size = int.from_bytes(size_payload, "little")
            #print(data_size)

            img_from_uart = ser.read(256)
            while len(img_from_uart) < data_size:
               img_from_uart += ser.read(256)
            #print(len(img_from_uart))

            nparr = np.frombuffer(img_from_uart, np.uint8)
            try:
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
                resized = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
                cv2.imshow('Image from GAP', resized)
                cv2.waitKey(1)
            except:
                print("Something went wrong with this image...")
       
    ser.close()

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
