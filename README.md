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
   * [RGB model](#rgb-model)
   * [BAYER model](#model-model)
 * [Additional features](#additional-features)
   * [Quantization](#quantization)

# Setup


In order to able to run th **inference on GVSOC**, one needs to *make sure* that GAP_SDK is installed. Once this requirement is satisfied, one needs to run the following command: 

```bash
source <GAP_SDK_HOME>/configs/gap9_v2.sh
```

In order to run **quantization part** of this repository, one needs to install some python dependencies. One sould keep in mind that some dependencies, such as `torch` and `torchvision` can be quite large. There we recommend to install following dependencies *only if* one wants to run the quantization part of this repository. 

```bash
pip install -r requirements.txt
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

| Model Ouput Description in `CI` mode |              |          
|---------------------------|-----------------------|
| extension                 | .bin                  |
| data type                 | float32               |
| output lenght             | 7 * `DO`<sup>3</sup>    |


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

## RGB model
To run GVSOC inference on default image run the following command:

```bash
cd inference_gvsoc
make all run platform=gvsoc
```

To run GVSOC inference on a different image, replace the image in 'inference_gvsoc_240x320_int/input.ppm' with the desired image. The image should be in `.ppm` format as described in the [table](#input-output-data-format) above. Then then rerun the command [above](#gvsoc-inference). 

## BAYER model

TODO: add description(When model will be merged)



# Additional features

## Training

Refer to the [Yolox repository](https://gitlab.com/xperience-ai/edge-devices/yolox/-/tree/yolox-master) for traning instructions. 

## Pytorch Inference

Refer to the [Demo](https://gitlab.com/xperience-ai/edge-devices/yolox/-/blob/yolox-master/tools/demo.py)

## ONNX conversion 

Refer to the [ONNX](https://gitlab.com/xperience-ai/edge-devices/yolox/-/blob/yolox-master/tools/export_onnx.py)


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




Follwing are the instructions on how to create dumps using GVSOC model.

1. Firstly you will need to create a folder with preprocessed images from the validation set. The folder should be located in one of the `inference_gvsoc_240x320_int` or `inference_gvsoc_240x320_int_bayer` folders, depending on the model you want to validate. In our case it is COCO2017 validations set. The format of the images should be `.ppm` for RGB and `.pgm` for BAYER respectively.

In our case we use COCO2017 validation set. If you need to use a different dataset you will need to inherit `CostomCOCODataset` class located in `quantization/utils.py` similar to how it is done in `GvsocInputGeneratorCOCO` class located in `quantization/utils.py`. Then run the following command:

Follwing are the instructions on how to create dumps using GVSOC model.

1. Firstly you will need to create a folder with preprocessed images from the validation set. The folder should be located in one of the `inference_gvsoc_240x320_int` or `inference_gvsoc_240x320_int_bayer` folders, depending on the model you want to validate. In our case it is COCO2017 validations set. The format of the images should be `.ppm` for RGB and `.pgm` for BAYER respectively.

In our case we use COCO2017 validation set. If you need to use a different dataset you will need to inherit `CostomCOCODataset` class located in `quantization/utils.py` similar to how it is done in `GvsocInputGeneratorCOCO` class located in `quantization/utils.py`. Then run the following command:

```bash
python quantization/gvsoc_gen_inputs.py --image_folder <path to folder with images>                 \
                                        --annotations <path to annotations>                         \
                                        --gvsoc_inputs <path to a folder to save inputs>            \
                                        --input_size <input size of the model>                      \
                                        --input_channels <number of input channels>                 \
                                        --model_type <model type for gvsoc validation>              \
```

2. Then you will need to create a folder for stroing the dumps inside one of the `inference_gvsoc_240x320_int` or `inference_gvsoc_240x320_int_bayer` folders, depending on the model you want to validate.

3. Finally you will need to run the following command:

```bash
cd <folder of the model you want to validate >
./val_ru.sh <input images folder name> <name of the folder for dumps>
```