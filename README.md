# YOLOX detection GAP inference

This repository contains the code for the inference of [YOLOX](https://arxiv.org/pdf/2107.08430.pdf)Nano on GVSOC.
The input is a 240x320 image and the output is the bounding box/es of detected people in the image. The model has been trained on the part of [CoCo dataset](https://cocodataset.org/#home) that contains only people. The model has a mAP of 0.3 on the val set of CoCo dataset. 



# Content 
 <!-- * [Requirements](#requirements) -->
 * [Setup](#Setup)
 * [Input & output data format](#Input-&-output-data-format)
 * [GVSOC Inference](#gvsoc)
 * [Additional features](#Additional-features)
   * [Quantization](#Quantization)



# Setup


In order to able to run th inference on GVSOC, one needs to *make sure* that GAP_SDK is installed. Once this requirement is satisfied, one needs to run the following command: 

```bash
source <GAP_SDK_HOME>/configs/gap9_v2.sh
```
# Input & output data format

<!-- create a table with the input and output data format -->

<!-- See the `audio` folder with examples of input data. -->
| Input Image Description   |          |
|---------------------------|----------|
| extension                 | .ppm     |
| data type                 | uint8    |
| image height              | 240      |
| image width               | 320      |
| number of image channels  | 3        |
| channel order             | BGR      |

| Model Ouput Description in `CI` mode |              |          
|---------------------------|-----------------------|
| extension                 | .bin                  |
| data type                 | float32               |
| output lenght             | 7 * DO<sup>1</sup>    |

[1]: DO is the number of Detected Objects. It is a variable that depends on the number of people in the input image. The maximum value of DO can not exceed `top_k_boxes` defined [here](./inference_gvsoc_240x320_int/main.h). Moreover one can choose the value of `top_k_boxes` according to use case. For examle if one is sure that maximux number of people in the input image will not exceed 50 then one can set `top_k_boxes` to 50. This will reduce meory usage.

The output of the model is a binary file with the name `output.bin` in the `out` directory. This file contains the bounding boxes of the detected people in the input image. The bounding boxes are represented in a sequence of 7 repeating elements. See the first 7 elements of the output file for an example:

| x1 | y1 | x2 | y2 | objectness score | class score | 1 |
|----|----|----|----|------------------|-------------|---|
|    |    |    |    |                  |             |   |

where `x1` and `y1` are the coordinates of the top left corner of the bounding box, `x2` and `y2` are the coordinates of the bottom right corner of the bounding box, `objectness score` is the score of the bounding box and `class score` is the score of the class of the detected object. The last column is always 1.


# GVSOC Inference

To run GVSOC inference on default image run the following command:

```bash
cd inference_gvsoc
make all run platform=gvsoc
```

To run GVSOC inference on a different image, replace the image in 'inference_gvsoc_240x320_int/input.ppm' with the desired image. The image should be in `.ppm` as described in the [table](#input--output-data-format) above. Then then rerun the command [above](#gvsoc-inference). 




# Additional features

## Quantization

If one wishes to quantize the model to 8 bit one can run the following command:

```bash

python quantization/quantize.py                                 \
        --path <path to .onnx model>                            \
        --coco_path <path to coco dataset>                      \
        --coco_annotations_path <path to annotations>           \
        --quant_dataset_size <size of the quantization dataset> \
        --input_size <input size of the model>                  \

```



