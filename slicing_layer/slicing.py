import numpy as np
import os
# from test_slicing import generate_test_data, pass_tests

def slicing_channel_fist(array):
    assert len(array.shape) == 3, "wrong input shape"
    c, h, w = array.shape

    #slicing here since delited it in the model
    patch_top_left = array[..., ::2, ::2]
    patch_top_right = array[..., ::2, 1::2]
    patch_bot_left = array[..., 1::2, ::2]
    patch_bot_right = array[..., 1::2, 1::2]
    array = np.concatenate(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        axis=0,
    )
    return array

def slicing_channel_last(array):
    assert len(array.shape) == 3, "wrong input shape"
    c, h, w = array.shape

    #slicing here since delited it in the model
    patch_top_left = array[::2, ::2, :]
    patch_top_right = array[::2, 1::2, :]
    patch_bot_left = array[1::2, ::2, :]
    patch_bot_right = array[1::2, 1::2, :]
    array = np.concatenate(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        axis=-1,
    )
    return array

def save_bin(array, path):
    with open(path, "wb") as in_f:
        array.astype(np.uint8, order='C', casting='unsafe', copy=True).tofile(in_f)


def read_bin(path):
    array = np.fromfile(path, dtype=np.uint8)
    return array


def slicing_chw_c_style(array, h, w, chnl):

    cout = 0 
    tmp = array.copy()
    # top left
    for c in range(0, chnl):
        for j in range(0, h // 2):
            k = j * (2 * w)  
            for i in range(0, w // 2): 
                # print("idx", k + i * 2)
                tmp[cout] = array[k + i * 2 + c *(w  * h)]
                # tmp[cout] = array[k + i * 2]
                cout += 1

    # # print("=")
    # # # bottom left
    for c in range(0, chnl):
        # print("c = ", c )
        for j in range(0, h // 2):
            k = j * (2 * w) + w  
            for i in range(0, w // 2): 
                # print("idx", k + i * 2 + c * (w * h))
                tmp[cout] = array[k + i * 2 + c *(w  * h)]
                cout += 1

    # # # top right 
    for c in range(0, chnl):
        for j in range(0, h // 2):
            k = j * (2 * w) + 1  
            for i in range(0, w // 2): 
                # print("idx", k + i * 2 + c * (w * h))
                tmp[cout] = array[k + i * 2 + c * (w * h)]
                cout += 1

    # # # bottom right
    for c in range(0, chnl):
        for j in range(0, h // 2):
            k = j * (2 * w) + w + 1 
            # print("idx", k + i * 2 + c * (w * h))
            for i in range(0, w // 2): 
                tmp[cout] = array[k + i * 2 + c * (w * h)]
                cout += 1
    return tmp

def slicing_hwc_c_style(array, h, w, chnl):

    cout = 0 
    tmp = array.copy()
    print(array.shape, w, h, chnl)

    # for col in range(h // 2):
        # for row in range(w // 2):
            # for c in range(chnl):
                # print(col*h*chnl + row*chnl + c, (2*col  )*w*chnl + (2*row)* chnl + c, sep='\t')
                # tmp[col*h*chnl + row*chnl + c] = array[(4*col)*w*chnl + (4*row)* chnl + c]

    

    if h > w : 
        split = (( w  * 3)) // (w // 2)  
    else:
        split = (h * ( w // 2 * 3)) // (w // 2) 

    split_cout = 0
    split_add = 0
    for i in range(0, (h * ( w // 2 * 3)), w // 2 * 3):
        for j in range(0, w):

            if j % 2 == 0:
                k = 3 * (j // 2)
            else:
                k = 3 * (w + j // 2)  

            if i >= split:

                if h > w:
                    split += (( w  * 3)) // (w // 2)  
                else:
                    split += split

                split_cout +=1
                split_add += (w * 3)

            k += i
            if h >= w: 
                k += split_add
            
            # print('===', k, array[k], i, j, split, split_cout, sep='\t')
      
            for c in range(0, chnl): 
                tmp[cout] = array[k + c]
                cout += 1
    return tmp


def generate_and_test():
    generate_test_data(low=1, high=255, path="./date/test_data_chw", size=10, channels=3, order='chw')
    pass_tests("./data/test_data_chw_source/")

if __name__ == "__main__":

    np.random.seed(1)
    generate_and_test()
    # generate_test_data(low=10, high=255, path="test_data", size=100, channels=3, order='chw')
    # pass_tests(source_path="./test_data_chw_source/")


    # array = np.arange(9,dtype=np.uint8).reshape(3, 3)[..., None]
    # print(array.shape)

    # # array = np.arange(4,dtype=np.uint8).reshape(2, 2)[..., None]
    # array = np.arange(16,dtype=np.uint8).reshape(4, 4)[...,None]
    # # array = np.arange(16,dtype=np.uint8).reshape(8, 2)[...,None]
    # # array = np.arange(16,dtype=np.uint8).reshape(2, 8)[...,None]
    # # array = np.arange(16,dtype=np.uint8).reshape(2, 8)[...,None]
    # # array = np.arange(36,dtype=np.uint8).reshape(6, 6)[..., None]
    # # array = np.arange(4 * 8, dtype=np.uint8).reshape(4, 8)[..., None]
    # # array = np.arange(4 * 8, dtype=np.uint8).reshape(8, 4)[..., None]
    # # array = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)[..., None]

    # # array = np.array([[1, 2], 
    # #                   [3, 4]], dtype=np.uint8)[..., None]
    # array = np.concatenate([array] * 3, axis=-1) 
    # h, w, c = array.shape

    # # array = np.array([[1, 2], 
    # #                   [3, 4]], dtype=np.uint8)[None]
    # # array = np.concatenate([array] * 3, axis=0) 
    # print(array)

    # print(array.shape)
    # save_bin(array, path="./mat_4_4.bin")
    # print(read_bin('./mat_4_4.bin'))


    # print(slicing_channel_last(array).shape)

    # save_bin(slicing_channel_last(array), path="./mat_4_4_python_sliced.bin")
    # print("sliced python version")
    # print(read_bin('./mat_4_4_python_sliced.bin'))

    # print("sliced C style version")
    # print(slicing_hwc_c_style(read_bin("./mat_4_4.bin"), h=h, w=w, chnl=c))


    # a = read_bin('./mat_4_4_python_sliced.bin')
    # b = slicing_hwc_c_style(read_bin("./mat_4_4.bin"), h=h, w=w, chnl=c)
    # print("===== mis match =====")
    # print(np.sum(a - b ))

    # # # print(array.shape)
    # # # print(array)
    # # save_bin(array, path="./test_data/mat_4_4.bin")
    # # # print(read_bin('./test_data/mat_42_594.bin'))

    # # # array_orig = read_bin('./test_1.bin')
    # # array_slicied = slicing(array)
    # # save_bin(array_slicied, path="./test_target/mat_4_4.bin")

    # # print("============== sliced correct ==================")
    # # array = slicing(array) 
    # # # print(array.shape) 
    # # # print(array) 
    # # save_bin(array)
    # # print(read_bin('./test_1.bin'))
