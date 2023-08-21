import numpy as np
import os
from time import time
from nntool.utils.fast_float import np_fastexp

def exp(x):
    return np.exp(x)


## naive implementation speed O(N * 6) and space O(1)
def decode_C_style(inputs, hw, strides):
    
    i = 0
    add = 0 
    for (h, w), stride in zip(hw, strides):
        size = (add + (h * w)) * 6
        grid1, grid2 = 0, -1 

        while i < size: 
            ## this operation costs a coutple of cycles
            if ((i // 6) % w) == 0:
                grid1 = 0
                grid2 += 1

            inputs[i] = (inputs[i]  + grid1 ) * stride
            inputs[i + 1] = (inputs[i + 1]  + grid2 ) * stride

            inputs[i + 2] = exp(inputs[i + 2]) * stride
            inputs[i + 3] = exp(inputs[i + 3]) * stride

            grid1 += 1
            i += 6

        add += h * w 

    return inputs


def decode_C_style_imp(inputs, hw, strides):

    i = 0
    for (height, width), stride in zip(hw, strides):
        # height, width = hw[j]
        for h in range(0, height):
            for w in range(0, width):
                inputs[i]     = (inputs[i    ] + w) * stride
                inputs[i + 1] = (inputs[i + 1] + h) * stride
                inputs[i + 2] = np_fastexp(inputs[i + 2]) * stride
                inputs[i + 3] = np_fastexp(inputs[i + 3]) * stride
                # print(f"[{i}] {inputs[i]:.2f}, {inputs[i+1]:.2f}, {inputs[i+2]:.2f}, {inputs[i+3]:.2f} {inputs[i+4]:.2f} {inputs[i+5]:.2f}")
                i += 6
    return inputs


def timeit(func, input, hw, strides, iters=100):
    ## calculate the average time and the standard deviation
    times = []
    for i in range(iters):
        start = time()
        func(input, hw, strides)
        end = time()
        times.append(end - start)
    print(f"{func.__name__} average time: {np.mean(times)} and std: {np.std(times)}")   

if __name__ == "__main__":

    strides = [8, 16, 32]
    # feature_map_sizes = [(8, 8), (4, 4), (2, 2)]
    feature_map_sizes = [(32, 40), (16, 20), (8, 10)]

    for file_name in os.listdir("./source"):  

        # read file outputs.bin
        with open(f"./source/{file_name}", "rb") as f:
            outputs = np.fromfile(f, dtype=np.float32)
            
        with open(f"./target/{file_name}", "rb") as f:
            target = np.fromfile(f, dtype=np.float32)

        timeit(decode_C_style, outputs.copy(), feature_map_sizes, strides)
        timeit(decode_C_style_imp, outputs.copy(), feature_map_sizes, strides)

        outputs_1 = decode_C_style(outputs.copy(), feature_map_sizes, strides)
        print(np.allclose(outputs_1, target)) 
        outputs_2 = decode_C_style_imp(outputs.copy(), feature_map_sizes, strides)
        print(np.allclose(outputs_2, target)) 


    print("\n\n\t\t =============== finished =============== \t\t")

    