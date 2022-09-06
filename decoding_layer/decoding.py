import numpy as np
from time import time


def exp(x, e=2.718281828459045):
    try:
        # return np.exp(x)
        # return e ** x
        output = e ** x
    except RuntimeWarning:
        print("overflow error for x ", x)
    return output


## naive implementation speed O(N * 6) and space O(1)
def decode_C_style(inputs, hw, strides):
    
    i = 0
    add = 0 
    for j in range(0, len(hw)):
        h, w = hw[j]
        size = (add + (h * w)) * 6
        grid1, grid2 = 0, -1 

        while i < size: 
            ## this operation costs a coutple of cycles
            if ((i // 6) % w) == 0:
                grid1 = 0
                grid2 += 1

            inputs[i] = (inputs[i]  + grid1 ) * strides[j] 
            inputs[i + 1] = (inputs[i + 1]  + grid2 ) * strides[j] 

            inputs[i + 2] = exp(inputs[i + 2]) * strides[j]
            inputs[i + 3] = exp(inputs[i + 3]) * strides[j] 

            grid1 += 1
            i += 6

        add += h * w 

    return inputs


def decode_C_style_imp(inputs, hw, strides):

    i = 0
    for j in range(0, len(hw)):
        height, width = hw[j]
        for h in range(0, height):
            for w in range(0, width):

                inputs[i] = (inputs[i] + w) * strides[j] 
                inputs[i + 1] = (inputs[i + 1] + h) * strides[j] 
                inputs[i + 2] = exp(inputs[i + 2]) * strides[j]
                inputs[i + 3] = exp(inputs[i + 3]) * strides[j] 

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

    ## read file outputs.bin
    with open("./source/data.bin", "rb") as f:
        outputs = np.fromfile(f, dtype=np.float32)
    print(outputs.shape)
    print(outputs[-10:])

    # with open("./targets.bin", "rb") as f:
    #     target = np.fromfile(f, dtype=np.float32)

    # timeit(decode_C_style, outputs.copy(), feature_map_sizes, strides)
    # timeit(decode_C_style_imp, outputs.copy(), feature_map_sizes, strides)

    # outputs_1 = decode_C_style(outputs.copy(), feature_map_sizes, strides)
    # print(np.allclose(outputs_1, target)) 
    # outputs_2 = decode_C_style_imp(outputs.copy(), feature_map_sizes, strides)
    # print(np.allclose(outputs_2, target)) 


    # print("finished")

    