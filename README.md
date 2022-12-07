# YOLOX detection GAP inference

This repository contains the code for the inference of [YOLOX](https://arxiv.org/pdf/2107.08430.pdf) Nano on GVSOC.
Also, it can be used to quantize both `RGB`<sup>1</sup> and `BAYER`<sup>2</sup> version of yolox model.


The input image resolution is QVGA and the output is the bounding box/es of detected people on the image. The model has been trained on the part of [COCO dataset](https://cocodataset.org/#home) that contains only people. The `RGB` model has a mAP of 0.3 on the val set of COCO dataset. 

[1] `RGB` version of yolox model is trained on the RGB images of COCO dataset.  


[2] `BAYER` version of yolox model is trained on the BAYER type of COCO dataset (see description about BAYER type of COCO dataset in [Quantization](###Quantization) part). Can be used to detected people on raw images from a camera with some accuracy loss.

# Content 
 <!-- * [Requirements](#requirements) -->
 * [Setup](#setup)
 * [Input & output data format](#input-&-output-data-format)
   * [RGB](#rgb)
   * [BAYER](#bayer)
   * [Output](#output)
 * [Performance](#performance)
 * [GVSOC Inference](#gvsoc-Inference)
   * [Inference Type Description](#inference-type-description)
   * [RGB model](#rgb-model)
   * [BAYER model](#model-model)
 * [Additional features](#additional-features)
   * [Quantization](#quantization)

# Setup

To start the porjec you will need: 

1. Python 3.10.4
2. Developer's GAP_SDK 


Execute the following commands to setup the project at the firt time: 

```bash
git clone git@gitlab.greenwaves-tech.com:Xperience/yolox_gap_inference.git 
cd yolox_gap_inference
virtualenv -p python3.10.4 venv
source venv/bin/activate
pip install -r requirements.txt
source <GAP_SDK_HOME>/configs/gap9_v2.sh (from drop down menu select GAP9_V2)
```

Next time you can just run the following commands to start the project: 

```bash
cd yolox_gap_inference
source venv/bin/activate
source <GAP_SDK_HOME>/configs/gap9_v2.sh (from drop down menu select GAP9_V2)
```

# Input & output data format

## RGB 
<!-- See the `audio` folder with examples of input data. -->

| Input Image Description   |          |
|---------------------------|----------|
| extension                 | .ppm     |
| data type                 | uint8    |
| image height              | 240      |
| image width               | 320      |
| number of image channels  | 3        |
| channel order             | RGB      |


## BAYER 

| Input Image Description   |          |
|---------------------------|----------|
| extension                 | .pgm     |
| data type                 | uint8    |
| image height              | 240      |
| image width               | 320      |
| number of image channels  | 1        |
| channel order             | RAW BAYER|

## Output

| Model Ouput Description in `CI` mode |              |          
|---------------------------|-----------------------|
| extension                 | .bin                  |
| data type                 | float32               |
| output lenght             | 7 * `DO`<sup>3</sup>    |

[3]: `DO` - Detected Objects. It is a variable that depends on the number of people in the input image. The maximum value of DO can not exceed `top_k_boxes` defined [here](./inference_gvsoc_240x320_int/main.h). Moreover one can choose the value of `top_k_boxes` according to use case. For examle if one is sure that maximux number of people in the input image will not exceed 50 then one can set `top_k_boxes` to 50. This will reduce meory usage.

The output of the model is a binary file with the name `output.bin` in the `out` directory. This file contains the bounding boxes of the detected people in the input image. The bounding boxes are represented in a sequence of 7 repeating elements. See the first 7 elements of the output file for an example:

| x1 | y1 | x2 | y2 | objectness score | class score | 1 |
|----|----|----|----|------------------|-------------|---|
|    |    |    |    |                  |             |   |

where `x1` and `y1` are the coordinates of the top left corner of the bounding box, `x2` and `y2` are the coordinates of the bottom right corner of the bounding box, `objectness score` is the score of the bounding box and `class score` is the score of the class of the detected object. The last column is always 1 indicating class `id`.



# Performance 


Following rusult are obtained using GVSOC inference and are validated on the RGB and BEYER version of COCO val2017 dataset respectively. 

| Model | Input resolution | mAP | AP@0.5 | AR@0.5| Gflops / GMac | Parameters (M) | Size (MB) |
|-------|------------------|-----|--------|-------|---------------|----------------|-----------|
| RGB   | 240x320          | 0.271 | 0.54 | 0.346 | 0.46 / 0.24   | 0.9            | 3,5       |
| BAYER | 240x320          | 0.229 | 0.46 | 0.309 | 0.42 / 0.22   | 0.9            | 3,5       |



# GVSOC Inference


## Inference Type Description

There are 3 types of inference that can be run on GVSOC. Each of them is described below.

- [+ DEMO +] In this model one can make inference on a single image.  The model will detect people on the image and draw bounding boxes around them. 

- [+ INFERENCE +] In thise model will only save detected bounding boxes in the output file.


## RGB model
To run GVSOC inference on default image run the following command:

```bash
cd inference_gvsoc_240x320_int
make all run platform=gvsoc mode=<choose the mode>
```

To run GVSOC inference on a different image, replace the image in 'inference_gvsoc_240x320_int/input.ppm' with the desired image. The image should be in `.ppm` format as described in the [table](#input-output-data-format) above. Then then rerun the command [above](#gvsoc-inference). 

## BAYER model

```bash

cd inference_gvsoc_240x320_int_bayer
make all run platform=gvsoc mode=<choose the mode>

```

To run GVSOC inference on a different image, replace the image in 'inference_gvsoc_240x320_int/input.pgm' with the desired image. The image should be in `.pgm` format as described in the [table](#input-output-data-format) above. Then then rerun the command [above](#gvsoc-inference). 


# Additional features

## Quantization

Originaly **RGB** model was quantized using 1000 random samples from [`COCO 2017` validation](http://images.cocodataset.org/zips/val2017.zip) set. The **BAYER** model however was quantized using `COCO 2017` which was converted to **synthetic BAYER** using the [ApproxVision repository](https://github.com/cucapra/approx-vision). What the repository does in a nutshell is it take a RGB image and convers it to BAYER image. The conversion is done using certain camera intrinsic parameters to reverse all ISP steps and in a way that the output image is as close as possible to the original image.

If one wishes to quantize the `RGB` model to 8 bit one can run the following command:

```bash

python quantization/quantize.py                                 \
        --path <path to .onnx model>                            \
        --coco_path <path to coco dataset>                      \
        --coco_annotations_path <path to annotations>           \
        --quant_dataset_size <size of the quantization dataset> \
        --input_size <input size of the model>                  \

```

if one wishes to quantize the `BAYER` model to 8 bit one can run the following command:

```bash

python quantization/quantize.py                                 \
        --path <path to .onnx model>                            \
        --coco_path <path to coco dataset>                      \
        --coco_annotations_path <path to annotations>           \
        --quant_dataset_size <size of the quantization dataset> \
        --input_size <input size of the model>                  \
        --input_channels 1                                      \

```




