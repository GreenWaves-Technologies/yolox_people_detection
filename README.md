# YOLOX detection GAP inference

This repository contains the code for the inference of [YOLOX](https://arxiv.org/pdf/2107.08430.pdf) Nano on GVSOC for people detection.

The input image resolution is QVGA and the output is the bounding box/es of the detected people on the image. The model has been trained on the part of [COCO dataset](https://cocodataset.org/#home) that contains only people. The `RGB` model has been trained on the COCO dataset and has a mAP of 0.3 on the val set of COCO dataset.

# Content 
 <!-- * [Requirements](#requirements) -->
 * [Setup](#setup)
 * [Input and output data format](#input-and-output-data-format)
   * [RGB](#rgb)
   * [Output](#output)
 * [Quantization](#quantization)
 * [NNTool Inference Mode](#nntool-inference-mode)
 * [Performance](#performance)
 * [GVSOC Inference](#gvsoc-Inference)
   * [RGB model](#rgb-model)
   * [BAYER model](#model-model)
 * [Additional features](#additional-features)

# Setup

In order to able to run th **inference on GVSOC**, one needs to *make sure* that GAP_SDK is installed. Once this requirement is satisfied, one needs to run the following command: 

```bash
source <GAP_SDK_HOME>/configs/gap9_v2.sh
```

In order to run **quantization part** of this repository, one needs to install some python dependencies.

```bash
pip install -r requirements.txt
```

# Input and output data format

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

| Model Ouput Description in `CI` mode |                       |          
|--------------------------------------|-----------------------|
| extension                            | .bin                  |
| data type                            | float32               |
| output lenght                        | 7 * `DO`<sup>3</sup>  |

<!-- 
## BAYER 

| Input Image Description   |          |
|---------------------------|----------|
| extension                 | .ppm     |
| data type                 | uint8    |
| image height              | 240      |
| image width               | 320      |
| number of image channels  | 3        |
| channel order             | RAW BAYER demosaiced|
-->

## Output

| Model Ouput Description in `CI` mode |                       |          
|--------------------------------------|-----------------------|
| extension                            | .bin                  |
| data type                            | float32               |
| output lenght                        | 7 * `DO`<sup>3</sup>  |

[3]: `DO` - Detected Objects. It is a variable that depends on the number of people in the input image. The maximum value of DO can not exceed `top_k_boxes` defined in [yolox.h](./yolox.h). Moreover one can choose the value of `top_k_boxes` according to use case. For example if one is sure that maximux number of people in the input image will not exceed 50 then one can set `top_k_boxes` to 50. This will reduce meory usage.

The output of the model is a binary file with the name `output.bin` in the `build` directory. This file contains the bounding boxes of the detected people in the input image. The bounding boxes are represented in a sequence of 7 repeating elements. See the first 7 elements of the output file for an example:

| x1   | y1  | x2    | y2     | objectness score | class score     | class ID |
|------|-----|-------|--------|------------------|-----------------|----------|
| left | top | right | bottom | bboxconfidence   | score det class | 1        |

where `x1` and `y1` are the coordinates of the top left corner of the bounding box, `x2` and `y2` are the coordinates of the bottom right corner of the bounding box, `objectness score` is the score of the bounding box and `class score` is the score of the class of the detected object. The last column is always 1 indicating class `id`.

# Quantization

Originaly **RGB** model was quantized using 1000 random samples from [`COCO 2017` validation](http://images.cocodataset.org/zips/val2017.zip) set:

```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

<!-- The **BAYER** model however was quantized using `COCO 2017` which was converted to **synthetic BAYER** using the [ApproxVision repository](https://github.com/cucapra/approx-vision). What the repository does in a nutshell is it take a RGB image and convers it to BAYER image. The conversion is done using certain camera intrinsic parameters to reverse all ISP steps and in a way that the output image is as close as possible to the original image. -->

In order to quantize the model in full int8, you need to collect statistics of all the NN layers. To do so you can run this command:

```bash

python deployment_scripts/quantize.py                           \
        --path <path to .onnx model>                            \
        --coco_path <path to coco dataset>                      \
        --coco_annotations_path <path to annotations>           \
        --quant_dataset_size <size of the quantization dataset> \
        --input_size <input size of the model>                  \

```

It will produce a pickle file (`precalculated_stats.pickle`) containing these statistics which can be used to prepare the graph by the nntool scripts. A precomputed version of this file for the `model/yolox_nano_rgb_240x320_wfocus.onnx` default model has been released alongside `model/precalculated_stats_yolox_nano_rgb_240x320_wfocus.pickle`.

In this project, the model has been quantized using HWC kernel layout and the NE16 quantization scheme. Look into the NNTool documentation for further information.

# NNTool Inference Mode

In this mode the application runs the model with one input image from file and writes the bounding boxes result to a binary file and the image (with bounding boxes). To change the image on which the model is run, you can change the variable `TEST_INPUT_FILE_NAME` in `CMakeLists.txt`.

The paths of the binary file containing the predicted bounding boxes and the image are defined in the `CMakeLists.txt` and by default are respectively `output.bin` and `output_compressed.jpg` in the CMake `build` directory. To further inspect the prediction you can visualize the binary bounding box into the image using `screen_scripts/show_boxes.py`:

```bash
python screen_scripts/show_boxes.py test_data/input.ppm build/output.bin
```

To check the correctness of the gvsoc/board inference you can use the python script in `deployment_scripts/nntool_script.py` in `inference` mode. It will execute the same model (quantized as for the gvsoc inference) and predict the bounding boxes.

Example of usage:

```bash
python deployment_scripts/nntool_script.py                                     \
      --mode="inference"                                                            \
      --trained_model weights/yolox_nano_rgb_240x320_wfocus.onnx                    \
      --stats_path weights/precalculated_stats_yolox_nano_rgb_240x320_wfocus.pickle \
      --input_image test_data/input.ppm
```

## CI mode

In CI mode, the `deployment_scripts/nntool_script.py` script is used to generate the ground truth bounding boxes which are then compared to the one predicted by Gvsoc/board.

# Performance 

## mAP

Following rusult are obtained using GVSOC inference and are validated on the RGB and BAYER version of COCO val2017 dataset respectively. 

| Model | Input resolution | mAP | AP@0.5 | AR@0.5| Gflops / GMac | Parameters (M) | Size (MB) |
|-------|------------------|-----|--------|-------|---------------|----------------|-----------|
| RGB   | 240x320          | 0.271 | 0.54 | 0.346 | 0.46 / 0.24   | 0.9            | 3,5       |
| BAYER | 240x320          | 0.229 | 0.46 | 0.309 | 0.42 / 0.22   | 0.9            | 3,5       |

## Cycles and Power consumption

The deployed model is capable of running more than 27 Operations per Cycle with an over all latency of less than 9MCyc. Here is the log from the execution of Inference mode:

```
                NN: Cycles:      8337365, Cyc%:  99.5%, Operations:    231800000, Op%: 100.0%, Operations/Cycle: 27.802549
          Decoding: Cycles:        40307, Cyc%:   0.3%, Operations:            0, Op%:   0.0%, Operations/Cycle: 0.000000
      Filter boxes: Cycles:        17603, Cyc%:   0.1%, Operations:            0, Op%:   0.0%, Operations/Cycle: 0.000000
        seq2bboxes: Cycles:          252, Cyc%:   0.0%, Operations:            0, Op%:   0.0%, Operations/Cycle: 0.000000
               NMS: Cycles:         1177, Cyc%:   0.0%, Operations:            0, Op%:   0.0%, Operations/Cycle: 0.000000
```

NOTE: in the INFERENCE mode also the JPEG compression is counted in the total cycles, but in real world use case, the image will be very rarely sent out, hence this extra time should not be taken into account.
NOTE2: Here the number of operation is counted with 1 MAC = 1 Op.

The overall power consumption of the GAP9 System depends on the operating mode:

| Model         | Quantization | GOps / GMac | Parameters (M) | Size (MB) | Cycles (M) | Op/Cyc | Energy per inference 370MHz@0.8V (Max Perf) (uJ) | Energy per inference 240MHz@0.65V (Max En Eff) (uJ) |
|---------------|--------------|-------------|----------------|-----------|------------|--------|--------------------------------------------------|-----------------------------------------------------|
| YoloxNano RGB | int8         | 0.46 / 0.24 | 0.9            | 0.9       |       8.65 | 53.18  |                                             1350 |                                                916  |


# GVSOC Inference
<!-- 
## RGB model
To run GVSOC inference on default image run the following command:

```bash
cd inference_gvsoc_240x320_int
rm -rf build weigths_tensors model.c modelInfos.h
cmake -B build -DMODE=<mode-name>
cd build
make make run -j  
```

To run GVSOC inference on a different image, replace the image in 'inference_gvsoc_240x320_int/input.ppm' with the desired image. The image should be in `.ppm` format as described in the [table](#input-output-data-format) above. Then then rerun the command [above](#gvsoc-inference). 

## BAYER model
To run GVSOC inference on default image run the following command:

```bash
cd inference_gvsoc_<model input size: 240x320 or 360x480>_bayer
rm -rf build weigths_tensors model.c modelInfos.h
cmake -B build -DMODE=<mode-name>
cd build
make make run -j  
``` -->

To run inference on default image run the following command:

```bash
cmake -B build -DCONFIG_<DEMO/INFERENCE>=y -DCONFIG_PLATFORM_<BOARD/GVSOC>=y
cmake --build build --target run -j
```

### Application MODEs:

  `DEMO` - Run the model in images coming from a camera, compress the images to jpeg and stream them via UART (works ONLY with platform=BOARD). Use the `demo_screen.py` python script to read the UART stream with your PC and display the images with bounding boxes predicted by GAP9. (NOT AVAILABLE on GVSOC)

  `INFERENCE` - Run the model on a single image (specified via Kconfig through the variable `CONFIG_IMAGE_REL_PATH`) and save the output in a file and visualized bounding boxes on the image. The image must be in `.ppm` format. 


# Additional features

## Training

Refer to the [Yolox repository](https://gitlab.com/xperience-ai/edge-devices/yolox/-/tree/yolox-master) for traning instructions. 

## Pytorch Inference

Refer to the [Demo](https://gitlab.com/xperience-ai/edge-devices/yolox/-/blob/yolox-master/tools/demo.py)

## ONNX conversion 

Refer to the [ONNX](https://gitlab.com/xperience-ai/edge-devices/yolox/-/blob/yolox-master/tools/export_onnx.py)
