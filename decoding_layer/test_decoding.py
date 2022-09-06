'''write unit that tests the function decode_C_style_imp
   using files from source and target folders'''

import os
import numpy as np
from decoding import decode_C_style_imp


def load_data(path):
    #open outputs.bin file
    with open(path) as in_f:
        ot = np.fromfile(in_f, dtype=np.float32)
    return ot 


def test(feateure_map_sizes, strides):

    ## list data in source folder
    source_files = sorted(os.listdir('source'))
    source_files = [os.path.join('source', f) for f in source_files]

    for sfile in source_files:
        tfile = sfile.replace('source', 'target')
        tfile = tfile.replace('outputs', 'targets')

        inputs = load_data(sfile)
        targets = load_data(tfile)

        outputs = decode_C_style_imp(inputs, feateure_map_sizes, strides)

        assert np.allclose(outputs, targets), "Test failed {}".format(sfile)
        print(f"Test passed for {sfile} \t +, the sum is {np.sum(abs(targets - outputs))}")



if __name__ == "__main__":

    ## do not forget to change the feateure_maps 
    ## according the the file you are testing

    # feature_maps = [(8, 8), (4, 4), (2, 2)]
    feature_maps = [(32, 40), (16, 20), (8, 10)]
    strides = [8, 16, 32]
    test(feature_maps, strides)