"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

from pathlib import Path, PureWindowsPath
import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image

import onnx
import onnxruntime



# Get x_scale_factor & y_scale_factor to resize image
def get_scale_factor(im_h, im_w, ref_size):

    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor

    
def predict_matte(im, ref_size, session):
    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size) 

    # resize image
    im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_CUBIC)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis = 0).astype('float32')
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_CUBIC)

    return matte


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir-path', type=str, help='path of the input image directory')
    # parser.add_argument('--output-path', type=str, help='paht for saving the predicted alpha matte (a file)')
    parser.add_argument('--model-path', type=str, help='path of the ONNX model')
    args = parser.parse_args()

    # check input arguments
    # if not os.path.exists(args.image_path):
    #     print('Cannot find the input image: {0}'.format(args.image_path))
    #     exit()
    # if not os.path.exists(args.model_path):
    #     print('Cannot find the ONXX model: {0}'.format(args.model_path))
    #     exit()

    ref_size = 1024

    # Initialize session and get prediction
    session = onnxruntime.InferenceSession(args.model_path, None)

    # ref = 256, 1024
    contrast_lut_lo = np.linspace(-10, 10, num=256, endpoint=True)
    contrast_lut_lo = ((1 / (1 + np.exp(1 * -contrast_lut_lo))) * 255).astype(np.uint8)
    contrast_lut_lo[contrast_lut_lo < 24] = 0
    contrast_lut_lo[contrast_lut_lo > 231] = 255

    from .curves import adjust_curves
    contrast_lut_hi = adjust_curves(np.arange(256, dtype=np.uint8), [[[7,0],[128,62],[221,227],[244,255]]])

    ##############################################
    #  Main Inference part
    ##############################################

    def create_matte(image_path):
        # read image
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # unify image channels to 3
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # normalize values to scale it between -1 to 1
        im = (im - 127.5) / 127.5   
    
        # contrast_lut_hi[contrast_lut_hi < 12] = 0
        # contrast_lut_hi[contrast_lut_hi > 243] = 255

        # for i in range(3,10):
        #     matte = predict_matte(im, int(128*i), session)
        #     matte = contrast_lut[matte]
        #     cv2.imwrite(f"{args.output_path}__mask_{int(128*i)}.PNG", matte)

        # 1024, 512
        matte_hi = predict_matte(im, 768, session)
        matte_hi = contrast_lut_hi[matte_hi]
        matte_hi = cv2.GaussianBlur(matte_hi, (5, 5), cv2.BORDER_DEFAULT)
        # matte_hi[matte_hi < 32] = 0
        # matte_hi[matte_hi > 224] = 255
        
        matte_lo = predict_matte(im, 256, session)
        matte_lo = contrast_lut_lo[matte_lo]
        # matte_lo[matte_lo < 32] = 0
        # matte_lo[matte_lo > 224] = 255
        # find edges, mask the middle tri-map (the fine hair area)
        mask_ref_sm = cv2.resize(matte_lo, None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_LINEAR)
        mask_grad_x = cv2.Sobel(mask_ref_sm, cv2.CV_16S, 0, 1, ksize=11, scale=100, delta=0, borderType=cv2.BORDER_DEFAULT)
        mask_grad_y = cv2.Sobel(mask_ref_sm, cv2.CV_16S, 1, 0, ksize=11, scale=100, delta=0, borderType=cv2.BORDER_DEFAULT)
        mask_abs_grad_x = cv2.convertScaleAbs(mask_grad_x)
        mask_abs_grad_y = cv2.convertScaleAbs(mask_grad_y)
        mask_border_sm = cv2.addWeighted(mask_abs_grad_x, 0.5, mask_abs_grad_y, 0.5, 0)
        mask_border = cv2.resize(mask_border_sm, dsize=matte_lo.shape[::-1], interpolation = cv2.INTER_LINEAR)
        mask_border_weight = mask_border / 255
        
        # final matte
        matte = np.clip((mask_border_weight * matte_hi) + ((1 - mask_border_weight) * matte_lo), 0, 255)

        return matte
    
    piped_data = sys.stdin.read()
    lines = piped_data.splitlines()
    paths = [Path(l.rstrip('\n')) for l in lines]

    for p in paths:
        # print(p.resolve())
        matte = create_matte(str(p.resolve()))
        output_path = p.parent / "mask" / (p.stem + '.PNG')
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path.resolve()), matte)
        print(f'saved {output_path}')

