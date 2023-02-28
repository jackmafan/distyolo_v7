import tensorrt as trt
import torch
import argparse
import os
import sys
import numpy as np

from yolo2onnx import LoadImage
# Reference 
# https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#build_engine_python
# https://blog.csdn.net/irving512/article/details/115403888
def trt_detect():

    # Load trt model
    logger = trt.Logger(trt.Logger.INFO)
    with open(opt.weights[0], 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # trt_model = TRTModule(engine, ['input'], ['output1', 'output2', '3', '4'])

    # Print model info
    model_all_names = []
    if opt.show_model:
        for idx in range(engine.num_bindings):
            is_input = engine.binding_is_input(idx)
            name = engine.get_binding_name(idx)
            op_type = engine.get_binding_dtype(idx)
            model_all_names.append(name)
            shape = engine.get_binding_shape(idx)
            print(f'input id:{idx} is input: {is_input} binding name :{name} shape:{shape} type {op_type}')

    # Load test image
    path, img, img0 = LoadImage(opt.test_input, opt.img_size, 32)
    tmp = img
    # Only for cuda:0 because is running on jetson nano
    img = torch.from_numpy(np.array([img])).to('cuda:0')
    img = img.float() # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # print(trt_model(img))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model.trt path')
    parser.add_argument('--test-input', default='', help='input image path')
    parser.add_argument('--img_size', default=640)
    parser.add_argument('--show-model', action='store_true', help='print details of trt model')
    opt = parser.parse_args()
    print(opt)

    trt_detect()