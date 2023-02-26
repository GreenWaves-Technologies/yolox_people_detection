import sys, serial, os, io
import PIL.Image as Image
import cv2

#Import all module inside tkinter
from tkinter import *
import tkinter.font as tkFont

#import pillow module and related module like
from PIL import Image,ImageTk
import numpy as np
from time import time


GAP_UART_BAUDRATE=3000000

INPUT_W=320
INPUT_H=240
PIXEL_SIZE=3

UART_START_JPEG=b'\xAB\xBA'


def receive_image(ser,l,t):
    count=0
    start = time()
    while True:
        read_bytes = ser.read(2)
        if read_bytes == UART_START_JPEG:
            size_payload = ser.read(1)
            while len(size_payload) < 4:
                size_payload += ser.read(1)
            data_size = int.from_bytes(size_payload, "little")
            #print(data_size)

            perf_array = ser.read(4*8)
            while len(perf_array) < 4*8:
                perf_array += ser.read(1)
            perf_array = np.frombuffer(perf_array, np.uint32)
            #print(perf_array)

            img_from_uart = ser.read(256)
            while len(img_from_uart) < data_size:
               img_from_uart += ser.read(256)
            #print(len(img_from_uart))

            nparr = np.frombuffer(img_from_uart, np.uint8)
            elapsed = time() - start
            try: 
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                resized = cv2.resize(img, (2*640, 2*480), interpolation = cv2.INTER_AREA)

                cv2.putText(resized, f'NN: {perf_array[:-1].sum()}us ({1e6/perf_array[:-1].sum():.2f}fps)',
                           (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                #cv2.putText(resized, f'[{1/elapsed:.2f}fps ({1e6/perf_array.sum():.2f}, {1e6/perf_array[:-1].sum():.2f})]',
                #           (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(resized, f'NN+JPEG: {perf_array.sum()}us ({1e6/perf_array.sum():.2f}fps)',
                            (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                #cv2.imshow('Image from GAP', resized)
                #cv2.waitKey(1)
            except:
                print("Something went wrong with this image...")

            resized_pil =Image.fromarray(resized)
            imgtk = ImageTk.PhotoImage(image=resized_pil)
            l.imgtk = imgtk
            l.configure(image=imgtk)

            count += 1
            start = time()
            break


    l.after(1,receive_image,ser,l,t)
    

def main():
    #Init the window otherwise the protocol get screwed
    #create a tkinter window
    t=Tk()
    t.geometry("1200x800")#here use alphabet 'x' not '*' this one
    #Create a label
    #app = Frame(t, bg="white")
    #app.grid()
    l=Label(t,font="bold", width=640, height=480)
    l.place(x=120,y=120,width=640, height=480)
    GLabel_413=Label(t)
    ft = tkFont.Font(family='Helvetica',size=30)
    GLabel_413["font"] = ft
    GLabel_413["fg"] = "#333333"
    GLabel_413["justify"] = "center"
    GLabel_413["text"] = "Greenwaves Technologies Smart Camera"
    GLabel_413.place(x=0,y=0,width=1200,height=120)
    #l.grid()
    l.pack(side = "bottom", fill = "both", expand = "yes")

    ser = serial.Serial(
        port='/dev/ttyUSB1',\
        baudrate=GAP_UART_BAUDRATE,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
            timeout=0.01)
    ser.flushInput()
    ser.flushOutput()
    print("connected to: " + ser.portstr)
    count=0
    
    l.after(20,receive_image,ser,l,t)
    t.mainloop() 
    

    ser.close()

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit