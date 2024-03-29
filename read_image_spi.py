import ft4222
from ft4222.SPI import Cpha, Cpol
from ft4222.SPIMaster import Mode, Clock, SlaveSelect
from ft4222.GPIO import Port, Dir
from time import sleep
import sys
from timeit import default_timer as timer
import PIL.Image as Image
import cv2
import numpy as np


#This chunk size should be the same of the one in C application
CHUNK_SIZE=256

magic_number    = b'\x47\x41'
magic_number_h  = 0x47
magic_number_l  = 0x41
cmd_status_get  = b'\xF1'
cmd_status_send = b'\xF2'
status_ready    = 0xD1
pad_value       = b'\x00'


def main():
    nbDev = ft4222.createDeviceInfoList()
    print("nb of fdti devices: {}".format(nbDev))

    ftDetails = []

    if nbDev <= 0:
        print("no devices found...")
        return

    print("devices:")
    for i in range(nbDev):
        detail = ft4222.getDeviceInfoDetail(i, False)
        print(" - {}".format(detail))
        ftDetails.append(detail)



    # open 'device' with default description 'FT4222'
    devA = ft4222.openByDescription('FT4222')

    # init spi master
    """ 
        Attributes:
            NONE:
            DIV_2: 1/2 System Clock
            DIV_4: 1/4 System Clock
            DIV_8: 1/8 System Clock
            DIV_16: 1/16 System Clock
            DIV_32: 1/32 System Clock
            DIV_64: 1/64 System Clock
            DIV_128: 1/128 System Clock
            DIV_256: 1/256 System Clock
            DIV_512: 1/512 System Clock

        NONE    = 0 // 60000000 Hz
        DIV_2   = 1 // 30000000 Hz
        DIV_4   = 2 // 15000000 Hz
        DIV_8   = 3 //  7500000 Hz
        DIV_16  = 4 //  3750000 Hz
        DIV_32  = 5 //  1875000 Hz
        DIV_64  = 6 //   937500 Hz
        DIV_128 = 7
        DIV_256 = 8
        DIV_512 = 9

    Cpha.CLK_LEADING  Cpha.CLK_TRAILING
    Cpol.IDLE_LOW, Cpol.IDLE_HIGH

    """

    # Clock on UMFT4222EV is at 60 MhZ 
    devA.spiMaster_Init(Mode.SINGLE, Clock.DIV_4,  Cpha.CLK_LEADING, Cpol.IDLE_LOW, SlaveSelect.SS0)

    # set port0 1 (-> note this is *not* the spi chip select, the chip select (SS0) is generated by the spi core)
    #devB.gpio_Write(Port.P0, 1)
    data_s = bytearray()
    data_r = bytearray()

    start = timer()
    while True:
        
        data_s = magic_number + cmd_status_get
        img_data   = bytearray()    
        #Inquiry device to see if it is ready to send data
        devA.spiMaster_SingleWrite(pad_value + data_s,True)
        data_r =devA.spiMaster_SingleRead(3+1,True)

        #print(hex(data_r[1]),hex(data_r[2]),hex(data_r[3]))

        if data_r[1] == magic_number_h and data_r[2] == magic_number_l and data_r[3] == status_ready:
            #read image size
            data_r =devA.spiMaster_SingleRead(4+1,True)
            image_size = (data_r[4] << 24) +(data_r[3] << 16) +(data_r[2] << 8) + data_r[1]
            #print(image_size)
            #read performance array
            data_r =devA.spiMaster_SingleRead(32+1,True)
            perf_0 = (data_r[4]  << 24) +(data_r[3]  << 16) +(data_r[2]  << 8) + data_r[1]
            perf_1 = (data_r[8]  << 24) +(data_r[7]  << 16) +(data_r[6]  << 8) + data_r[5]
            perf_2 = (data_r[12] << 24) +(data_r[11] << 16) +(data_r[10] << 8) + data_r[9]
            perf_3 = (data_r[16] << 24) +(data_r[15] << 16) +(data_r[14] << 8) + data_r[13]
            perf_4 = (data_r[20] << 24) +(data_r[19] << 16) +(data_r[18] << 8) + data_r[17]
            perf_5 = (data_r[24] << 24) +(data_r[23] << 16) +(data_r[22] << 8) + data_r[21]
            perf_6 = (data_r[28] << 24) +(data_r[27] << 16) +(data_r[26] << 8) + data_r[25]
            perf_7 = (data_r[32] << 24) +(data_r[31] << 16) +(data_r[30] << 8) + data_r[29]

            read_size=0
            remaining_size = image_size
            while remaining_size > 0:
                if remaining_size > CHUNK_SIZE:
                    read_size = CHUNK_SIZE
                else:
                    read_size = remaining_size
                
                tmp_data  =devA.spiMaster_SingleRead(read_size+1,True)
                img_data +=tmp_data[1:read_size+1]
                remaining_size-=read_size

            nparr = np.frombuffer(img_data, np.uint8)            
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1


            img_resized = cv2.resize(img, (int(1.5*640), int(1.5*480)), interpolation = cv2.INTER_AREA)
            rect = np.zeros(( 90,int(1.5*640),3),np.uint8)
            rect[:] = 255 
            vis = np.concatenate((rect, img_resized), axis=0)
            
            nn_perf = perf_0+perf_1+perf_2+perf_3+perf_4+perf_5+perf_6
            full_perf = nn_perf+perf_7
            #cv2.rectangle(resized, (0, 0), (960, 40), (255,255,255), 150)
            cv2.putText(vis, f'NN: {round(nn_perf/1000,1)}ms ({1e6/nn_perf:.2f}fps) - 0.85 mJoule/frame',
                       (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)
            #cv2.putText(resized, f'[{1/elapsed:.2f}fps ({1e6/perf_array.sum():.2f}, {1e6/perf_array[:-1].sum():.2f})]',
            #           (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
            cv2.putText(vis, f'NN+JPEG: {round(full_perf/1000,1)}ms ({1e6/full_perf:.2f}fps)',
                            (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)
                
            # #im = Image.frombuffer('L',(im_width,im_height),data,'raw','L',0,1)
            # #im = Image.open(data)
            # flatNumpyArray = numpy.array(img_data,dtype=numpy.uint8)

            # open_cv_image = flatNumpyArray.reshape(im_height, im_width)
            # bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_BAYER_RG2BGR)

            # ## In case of FULL HD reduce size 
            # if im_width == 2688:
            #     bgr = cv2.resize(bgr, (960,540), interpolation = cv2.INTER_AREA)    
            # #bgr = cv2.resize(bgr, (640,480), interpolation = cv2.INTER_AREA)
            cv2.imshow('Gap Output',vis)
            cv2.waitKey(1)
            # end = timer()
            # ttime=(end - start)
            # fps=1/ttime
            # print('Transfer time: ',round(ttime,2),'FPS: ',round(fps,1),'\t Bandwidth: ',round(((im_width*im_height)/ttime)/1000000,3),'MBytes/s')
            # start = timer()

if __name__ == "__main__":
    sys.exit(main())