import numpy as np
from utils import read_bin, save_bin

def slicing_channel_fist(array):
    assert len(array.shape) == 3, "wrong input shape"

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

    patch_top_left = array[::2, ::2, :]
    patch_bot_left = array[1::2, ::2, :]
    patch_top_right = array[::2, 1::2, :]
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

def slicing_chw_c_style(array, h, w, chnl):

    cout = 0 
    tmp = array.copy()
    # top left
    for c in range(0, chnl):
        for j in range(0, h // 2):
            k = j * (2 * w)  
            for i in range(0, w // 2): 
                tmp[cout] = array[k + i * 2 + c *(w  * h)]
                cout += 1

    # bottom left
    for c in range(0, chnl):
        # print("c = ", c )
        for j in range(0, h // 2):
            k = j * (2 * w) + w  
            for i in range(0, w // 2): 
                tmp[cout] = array[k + i * 2 + c *(w  * h)]
                cout += 1

    # top right 
    for c in range(0, chnl):
        for j in range(0, h // 2):
            k = j * (2 * w) + 1  
            for i in range(0, w // 2): 
                tmp[cout] = array[k + i * 2 + c * (w * h)]
                cout += 1

    # bottom right
    for c in range(0, chnl):
        for j in range(0, h // 2):
            k = j * (2 * w) + w + 1 
            for i in range(0, w // 2): 
                tmp[cout] = array[k + i * 2 + c * (w * h)]
                cout += 1
    return tmp


def slicing_hwc_c_style(array, h, w, channels):

    tmp_array = array.copy()
    cur = 0
    tmp1 = 0 
    tmp2 = w * channels 
    comp = (w * h * channels) // (h // 2) 

    larger, smaller = (h, w) if h > w else (w, h)

    for j in range(0, larger):
        for i in range(0, smaller):
            for c in range(0, channels):
                if i % 2 == 0:                
                    tmp_array[cur] = array[tmp1]
                    tmp1 += 1
                else:
                    tmp_array[cur] = array[tmp2]
                    tmp2 += 1 

                cur += 1
                if  cur % comp == 0: 
                    tmp1 += w * channels
                    tmp2 += w * channels
                
    return tmp_array


def slicing_hwc_c_style_imp(array, h, w, channels):

    o_h, o_w = h // 2, w // 2
    o_c = channels * 4
    output = np.zeros((o_h * o_w * o_c ), dtype=np.uint8)

    for j in range(0, o_h):
        for i in range(0, o_w):
            for c in range(0, channels): 
                output[j * o_w * o_c + i * o_c + c]     = array[(j * 2 * w * channels) + (i * 2 * channels) + c]
                output[j * o_w * o_c + i * o_c + 3 + c] = array[(j * 2 * w * channels) + (i * 2 * channels) + (channels * w) +  c]
                output[j * o_w * o_c + i * o_c + 6 + c] = array[(j * 2 * w * channels) + (i * 2 * channels) + channels  + c]
                output[j * o_w * o_c + i * o_c + 9 + c] = array[(j * 2 * w * channels) + (i * 2 * channels) + (channels * w + channels)  + c]
    return output 

def check_idex(input_array, input_seq, h, w, channels): 

    o_h, o_w = h // 2, w // 2
    o_c = channels * 4
    print("input array shape", input_array.shape)
    output = np.zeros((o_h, o_w, o_c), dtype=np.uint8)
    output_seq = np.zeros((input_seq.shape), dtype=np.uint8)

    for j in range(0, o_h):
        for i in range(0, o_w):
            for c in range(0, channels): 

                output[j, i, c]     = input_array[j * 2, i * 2, c]
                output[j, i, 3 + c] = input_array[j * 2 + 1, i * 2, c]
                output[j, i, 6 + c] = input_array[j * 2, i * 2 + 1, c]
                output[j, i, 9 + c] = input_array[1 + j * 2, 1 + i * 2, c]
            
                output_seq[j * o_w * o_c + i * o_c + c]     = input_seq[(j * 2 * w * channels) + (i * 2 * channels) + c]
                output_seq[j * o_w * o_c + i * o_c + 3 + c] = input_seq[(j * 2 * w * channels) + (i * 2 * channels) + (channels * w) +  c]
                output_seq[j * o_w * o_c + i * o_c + 6 + c] = input_seq[(j * 2 * w * channels) + (i * 2 * channels) + channels  + c]
                output_seq[j * o_w * o_c + i * o_c + 9 + c] = input_seq[(j * 2 * w * channels) + (i * 2 * channels) + (channels * w + channels)  + c]


                # output_seq[j * o_w * 12 + i * 12 + c] = input[j * 2 * w * channels + i * 2 * channels + c]
                # print(f"idxOutImp = {j * o_w * 12 + i * 12 + c:>2}\tidxOutTrue = {j * 2 * w * channels + i * 2 * channels + c:>2}\ti = {i:>2}\tj = {j:>2}")
                # idxInp = j * o_w * channels + i * channels + c
                # print(f"idxOutImp = {j * o_w * 12 + i * 12 + c:>2}\tidxOutTrue = {j * 2 * w * channels + i * 2 * channels + c:>2}\ti = {i:>2}\tj = {j:>2}")
                # print(f"idxOutImp = {j * o_w * 12 + 3  + i * 12 + 3 + c:>2}\tidxOutTrue = {(1 + j * 2 * w * channels) + i * 2 * channels + c:>2}\ti = {i:>2}\tj = {j:>2}")
                # print(f"idxOutImp = {(j * o_w * 12 + 6)  + (i * 12 + 6) + c:>2}\tidxOutTrue = {(j * 2 * w * channels + 1) + (i * 2 * channels) + c:>2}\ti = {i:>2}\tj = {j:>2}")
                # print(f"idxOutImp = {(j * o_w * 12 + 9)  + (i * 12 + 9) + c:>2}\tidxOutTrue = {(1 + j * 2 * w * channels) + (1 + i * 2 * channels) + c:>2}\ti = {i:>2}\tj = {j:>2}")


    input_seq = input_seq.reshape((h, w, channels))
    output_seq = output_seq.reshape((o_h, o_w, o_c))

    cor_output = slicing_channel_last(input_array)
    print("correct output")
    # print(input_array)
    print("my output")
    print(output)
    print("=====")

    print("correct output")
    # print(input_seq)
    print("my output")
    print(output_seq)
    # print("diff sum", np.sum(cor_output - output))
    # print("diff sum", np.sum(cor_output - output))
    print("diff")
    ## flatten and compare
    # print(np.sum(cor_output.flatten() - output.flatten()))
    print(np.sum(output - output_seq))

def slicing_hw1_style(array, h, w):

    tmp_array = array.copy()
    cur = 0
    tmp1 = 0 
    tmp2 = w 
    comp = (w * h) // (h // 2) 
    larger, smaller = (h, w) if h > w else (w, h)

    for j in range(0, larger):
        for i in range(0, smaller):
            if i % 2 == 0:                
                tmp_array[cur] = array[tmp1]
                tmp1 += 1
            else:
                tmp_array[cur] = array[tmp2]
                tmp2 += 1 

            cur += 1
            if  cur % comp == 0: 
                tmp1 += w
                tmp2 += w

    return tmp_array

if __name__ == "__main__":

    np.random.seed(1)

    ##### main part #####
    # h = 4
    # w = 2
    # c = 3

    # array = np.arange(h * w * c).reshape(h, w, c)
    # save_bin(array, f"array_{w}_{h}_{c}.bin")
    # array_python = read_bin(f"array_{w}_{h}_{c}.bin")
    # print(array_python)


    # array_scliced_python = slicing_channel_last(array)
    # save_bin(array_scliced_python, f"array_scliced_{w}_{h}_{c}.bin")
    # array_sliced_bin = read_bin(f"array_scliced_{w}_{h}_{c}.bin")
    # print(array_sliced_bin)

    # for h in range(2, 400, 2 ):
    #     for w in range(4, 400, 2):
    h = 240  
    w = 320 
    # h = 4
    # w = 4
    c = 3
    array = np.arange(h * w * c, dtype=np.uint64).reshape(h, w, c)
    # array = np.random.randint(low=0, high=255, size=(h * w * c), dtype=np.uint8).reshape(h, w, c )
    
    # test the dataset on the slicing function
    h, w, c = array.shape
    print(array)
    print(array.shape)

    save_bin(array, "./array.bin")
    array_python = read_bin("./array.bin")

    print(array_python)
    print(array_python.shape)

    array_scliced_python = slicing_channel_last(array)

    print(array_scliced_python)
    print(array_scliced_python.shape)

    save_bin(array_scliced_python, "array_scliced_python.bin")
    array_sliced_bin = read_bin("array_scliced_python.bin")


    print(array_sliced_bin)
    print(array_sliced_bin.shape)

    # call the fucntion 
    sliced_c_stype = slicing_hwc_c_style_imp(array_python, h, w, channels=c) 

    mistach = np.sum(sliced_c_stype - array_sliced_bin)
    print(f"mistach = {mistach} w = {w} h = {h}")
    assert mistach == 0, f"mismatch: {mistach} h={h} w={w} c={c}"
            