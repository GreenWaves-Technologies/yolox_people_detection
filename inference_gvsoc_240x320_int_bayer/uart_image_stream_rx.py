import sys, serial, os, io
import PIL.Image as Image
import cv2, numpy


#INPUT_W=640
#INPUT_H=480

INPUT_W=320
INPUT_H=240
PIXEL_SIZE=1

UART_START_COM=b'\xF1\x1F'

DEBAYER=True

def main():
    #Init the window otherwise the protocol get screwed
    open_cv_image = numpy.zeros(shape=(INPUT_H, INPUT_W))
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
        if read_bytes == UART_START_COM:
            out = ser.read(INPUT_W*PIXEL_SIZE)
            while len(out) < INPUT_H*INPUT_W*PIXEL_SIZE :
                out += ser.read(INPUT_W*PIXEL_SIZE)
            #if len(out) < 320*240:
            #out += ser.read(320*240-len(out))
            if PIXEL_SIZE == 1:
                im = Image.frombuffer('L',(INPUT_W,INPUT_H),out,'raw','L',0,1)
            elif PIXEL_SIZE == 2:
                im = Image.frombuffer('I;16',(INPUT_W,INPUT_H),out,'raw','L',0,1)
            #im.save("received_img/img_"+str(count)+".png")
            open_cv_image = numpy.array(im)
            if DEBAYER:
                #cv2.imwrite("saved_images/"+str(count)+".pgm",open_cv_image)
                bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_BAYER_GB2BGR)
                resized = cv2.resize(bgr, (INPUT_W*2,INPUT_H*2), interpolation = cv2.INTER_AREA)
            else:
                resized = cv2.resize(open_cv_image, (INPUT_W,INPUT_H), interpolation = cv2.INTER_AREA)
            cv2.imshow('Image from GAP',resized)
            cv2.waitKey(1)
            #print(count)
            count=count+1
       
    ser.close()

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit