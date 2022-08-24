import numpy as np

def save_bin(array, path):
    with open(path, "wb") as in_f:
        array.astype(np.uint8, order='C', casting='unsafe', copy=True).tofile(in_f)


def read_bin(path):
    array = np.fromfile(path, dtype=np.uint8)
    return array