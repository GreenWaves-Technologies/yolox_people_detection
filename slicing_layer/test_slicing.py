import os
import numpy as np
from slicing import slicing_chw_c_style, slicing_hwc_c_style
from slicing import slicing_channel_fist, slicing_channel_last
from utils import save_bin, read_bin

def generate_test_data(low, high, path, size=1, channels=3, order='chw'):

    assert channels > 0, "wrong input channels, should be > 0"
    assert low > 0 and high > 0, "wrong range of input dimensions, low > 0, high > 0"
    assert order in ['chw', 'hwc'], "wrong order, choose from[chw, hwc]"
    sequence = [i for i in range(low, high) if i % 2 == 0]


    if os.path.exists(path + f"_{order}_source"):
        #remove all files in the folder
        for file in os.listdir(path + f"_{order}_source"): 
            os.remove(path + f"_{order}_source/" + file) 
    else:
        os.makedirs(path + f"_{order}_source")

    if os.path.exists(path + f"_{order}_target"):
        #remove all files in the folder
        for file in os.listdir(path + f"_{order}_target"): 
            os.remove(path + f"_{order}_target/" + file) 
    else:
        os.makedirs(path + f"_{order}_target")


    for i in range(size): 

        h = np.random.choice(sequence, size=1)[0]
        w = np.random.choice(sequence, size=1)[0]

        if order == 'chw':
            array = np.random.randint(low=0, high=255, size = (channels, h, w), dtype=np.uint8)
            save_bin(array, path=f"./{path}_{order}_source/mat_{channels}_{h}_{w}.bin")
        else:
            array = np.random.randint(low=0, high=255, size = (h, w, channels), dtype=np.uint8)
            save_bin(array, path=f"./{path}_{order}_source/mat_{h}_{w}_{channels}.bin")

        if order == 'chw':
            array = slicing_channel_fist(array)
            save_bin(array, path=f"./{path}_{order}_target/mat_{channels}_{h}_{w}.bin")
        else:
            array = slicing_channel_last(array)
            save_bin(array, path=f"./{path}_{order}_target/mat_{h}_{w}_{channels}.bin")

def pass_tests(source_path):

    assert 'chw' in source_path or 'hwc' in source_path, "wrong should be in [chw, hwc]"
    order = 'chw' if 'chw' in source_path else 'hwc'
    target_path = source_path.replace('source', 'target')

    for file in os.listdir(source_path):
        if order == 'chw':
            c, h, w = [int(i) for i in file.split(".")[0].split("_")[1:]]
        else:
            h, w, c = [int(i) for i in file.split(".")[0].split("_")[1:]]

        array = read_bin(source_path + file)
        array_target = read_bin(target_path + file)
        if order == 'chw':
            array_sliced_c_style = slicing_chw_c_style(array, h, w, c)
        else:
            array_sliced_c_style = slicing_hwc_c_style(array, h, w, c)

        equal = np.allclose(array_target, array_sliced_c_style)
        print(f"test - {file}:   \t +") if equal else print(f"test - {file}:   \t -")

def generate_and_test(**kwargs):
    generate_test_data(
        path="./data/test", low=kwargs["low"],
        high=kwargs["high"], size=kwargs["size"], 
        channels=kwargs["channels"], order=kwargs["order"])

    pass_tests(source_path="./data/test_chw_source/")

if __name__ == "__main__":
    generate_and_test(size=30, low=10, high=100, channels=3, order='chw')