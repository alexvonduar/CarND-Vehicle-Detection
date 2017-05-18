import cv2
import numpy as np

def color_convert(img, src_cspace, dst_cspace):
    if src_cspace == 'RGB':
        if dst_cspace == 'HSV':
            converter = cv2.COLOR_RGB2HSV
        elif dst_cspace == 'LUV':
            converter = cv2.COLOR_RGB2LUV
        elif dst_cspace == 'HLS':
            converter = cv2.COLOR_RGB2HLS
        elif dst_cspace == 'YUV':
            converter = cv2.COLOR_RGB2YUV
        elif dst_cspace == 'YCrCb':
            converter = cv2.COLOR_RGB2YCrCb
        elif dst_cspace == 'BGR':
            converter = cv2.COLOR_RGB2BGR
        elif dst_cspace == 'GRAY':
            converter = cv2.COLOR_RGB2GRAY
        elif dst_cspace == 'RGB':
            converter = None
        else:
            print("wrong dst color space:", dst_cspace)
            exit(1)
    elif src_cspace == 'BGR':
        if dst_cspace == 'HSV':
            converter = cv2.COLOR_BGR2HSV
        elif dst_cspace == 'LUV':
            converter = cv2.COLOR_BGR2LUV
        elif dst_cspace == 'HLS':
            converter = cv2.COLOR_BGR2HLS
        elif dst_cspace == 'YUV':
            converter = cv2.COLOR_BGR2YUV
        elif dst_cspace == 'YCrCb':
            converter = cv2.COLOR_BGR2YCrCb
        elif dst_cspace == 'RGB':
            converter = cv2.COLOR_BGR2RGB
        elif dst_cspace == 'GRAY':
            converter = cv2.COLOR_BGR2GRAY
        elif dst_cspace == 'BGR':
            converter = None
        else:
            print("Wrong dst color space:", dst_cspace)
            exit(1)
    else:
        print("input color space must be RGB or BGR:", src_cspace)
        exit(1)

    if converter != None:
        image = cv2.cvtColor(img, converter)
    else:
        image = np.copy(img)

    return image

def get_cmap(cspace):
    if cspace == 'HSV':
        cmap = 'hsv'
    elif cspace == 'LUV':
        cmap = 'luv'
    elif cspace == 'HLS':
        cmap = 'hls'
    elif cspace == 'YUV':
        cmap = 'yuv'
    elif cspace == 'YCrCb':
        cmap = 'ycrcb'
    elif cspace == 'BGR':
        cmap = 'bgr'
    elif cspace == 'RGB':
        cmap = 'rgb'
    else:
        print("color space not supported:", cspace)
        cmap = 'rgb'

    return cmap
