# Distyolo on yolov7

Use image to predict the distance with modified structure of yolov7

## Reference thesis:
https://www.mdpi.com/2076-3417/12/3/1354

## Environment:
*   torch 1.8.0
*   torchvision 0.9.0
*   python 3.6.9

## onnx -> tensorrt
```bash
    $ source onnx2trt.sh best-nms.onnx best.trt
```