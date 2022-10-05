import numpy as np

if __name__ == '__main__':

    vis_size = 10
    #Load the input data
    input_with_python = np.fromfile("./Input_1_python_sliced.bin", dtype=np.int8)
    input_with_c = np.fromfile("./Input_1_original.bin", dtype=np.int8)
    # Compare the inputs
    print("Original input: \t ", input_with_c[:vis_size])
    print("Sliced input: \t ", input_with_python[:vis_size])
    print("=== The inputs shoud not be equal ===")
    res =  np.sum(input_with_python - input_with_c)
    assert res != 0, "The inputs are not equal"
    print(res)

    # Load the output data 
    print("==== The outputs comparision ====")
    output_with_python_slicing = np.fromfile("./Output_1_python_sliced.bin", dtype=np.int8)
    output_with_c_slicing = np.fromfile("./Output_1_C_sliced.bin", dtype=np.int8)
    output_onnx = np.fromfile("./Output_1_onnx.bin", dtype=np.int8)

    # Compare the outputs   
    print("Python output: \t", output_with_python_slicing[:vis_size]) 
    print("C output: \t", output_with_c_slicing[:vis_size])
    print("Onnx output: \t", output_onnx[:vis_size])
    print('=== The outputs should be equal ===')
    res = np.sum(output_with_python_slicing - output_with_c_slicing)
    assert res == 0, "The outputs are not equal" 
    print(np.sum(output_with_python_slicing - output_with_c_slicing))