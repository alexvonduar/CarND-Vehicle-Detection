import cv2
import numpy as np


def convert_from_rgb(img, dcspace):
    '''convert rgb image to destination color space'''
    target_color_space = dcspace
    if dcspace == 'HSV':
        converter = cv2.COLOR_RGB2HSV
    elif dcspace == 'LUV':
        converter = cv2.COLOR_RGB2LUV
    elif dcspace == 'HLS':
        converter = cv2.COLOR_RGB2HLS
    elif dcspace == 'YUV':
        converter = cv2.COLOR_RGB2YUV
    elif dcspace == 'YCrCb':
        converter = cv2.COLOR_RGB2YCrCb
    elif dcspace == 'BGR':
        converter = cv2.COLOR_RGB2BGR
    elif dcspace == 'GRAY':
        converter = cv2.COLOR_RGB2GRAY
    elif dcspace == 'RGB':
        converter = None
    else:
        print("wrong dst color space:", dcspace)
        converter = None
        target_color_space = 'RGB'

    if converter != None:
        return cv2.cvtColor(img, converter), target_color_space
    else:
        return img, target_color_space


def convert_from_bgr(img, dcspace):
    '''convert image from BGR color space to destination color space'''
    target_color_space = dcspace
    if dcspace == 'HSV':
        converter = cv2.COLOR_BGR2HSV
    elif dcspace == 'LUV':
        converter = cv2.COLOR_BGR2LUV
    elif dcspace == 'HLS':
        converter = cv2.COLOR_BGR2HLS
    elif dcspace == 'YUV':
        converter = cv2.COLOR_BGR2YUV
    elif dcspace == 'YCrCb':
        converter = cv2.COLOR_BGR2YCrCb
    elif dcspace == 'RGB':
        converter = cv2.COLOR_BGR2RGB
    elif dcspace == 'GRAY':
        converter = cv2.COLOR_BGR2GRAY
    elif dcspace == 'BGR':
        converter = None
    else:
        print("Wrong dst color space:", dcspace)
        converter = None
        target_color_space = 'BGR'

    if converter != None:
        return cv2.cvtColor(img, converter), target_color_space
    else:
        return img, target_color_space


def convert_from_hsv(img, dcspace):
    ''' convert image from HSV color space into destination color space'''
    target_color_space = dcspace
    if dcspace == 'HSV':
        converter = None
    elif dcspace == 'LUV':
        converter = cv2.COLOR_HSV2LUV
    elif dcspace == 'HLS':
        converter = cv2.COLOR_HSV2HLS
    elif dcspace == 'YUV':
        converter = cv2.COLOR_HSV2YUV
    elif dcspace == 'YCrCb':
        converter = cv2.COLOR_HSV2YCrCb
    elif dcspace == 'RGB':
        converter = cv2.COLOR_HSV2RGB
    elif dcspace == 'GRAY':
        converter = cv2.COLOR_HSV2GRAY
    elif dcspace == 'BGR':
        converter = cv2.COLOR_HSV2BGR
    else:
        print("Wrong dst color space:", dcspace)
        converter = None
        target_color_space = 'HSV'

    if converter != None:
        return cv2.cvtColor(img, converter), target_color_space
    else:
        return img, target_color_space


def convert_from_hls(img, dcspace):
    ''' convert image from HLS color space into destination color space'''
    target_color_space = dcspace
    if dcspace == 'HSV':
        converter = cv2.COLOR_HLS2HSV
    elif dcspace == 'LUV':
        converter = cv2.COLOR_HLS2LUV
    elif dcspace == 'HLS':
        converter = None
    elif dcspace == 'YUV':
        converter = cv2.COLOR_HLS2YUV
    elif dcspace == 'YCrCb':
        converter = cv2.COLOR_HLS2YCrCb
    elif dcspace == 'RGB':
        converter = cv2.COLOR_HLS2RGB
    elif dcspace == 'GRAY':
        converter = cv2.COLOR_HLS2GRAY
    elif dcspace == 'BGR':
        converter = cv2.COLOR_HLS2BGR
    else:
        print("Wrong dst color space:", dcspace)
        converter = None
        target_color_space = 'HLS'

    if converter != None:
        return cv2.cvtColor(img, converter), target_color_space
    else:
        return img, target_color_space

def convert_from_yuv(img, dcspace):
    ''' convert image from YUV color space into destination color space'''
    target_color_space = dcspace
    if dcspace == 'HSV':
        converter = cv2.COLOR_YUV2HSV
    elif dcspace == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        converter = cv2.COLOR_RGB2LUV
    elif dcspace == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        converter = cv2.COLOR_RGB2HLS
    elif dcspace == 'YUV':
        converter = None
    elif dcspace == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        converter = cv2.COLOR_RGB2YCrCb
    elif dcspace == 'RGB':
        converter = cv2.COLOR_YUV2RGB
    elif dcspace == 'GRAY':
        converter = cv2.COLOR_YUV2GRAY
    elif dcspace == 'BGR':
        converter = cv2.COLOR_YUV2BGR
    else:
        print("Wrong dst color space:", dcspace)
        converter = None
        target_color_space = 'YUV'

    if converter != None:
        return cv2.cvtColor(img, converter), target_color_space
    else:
        return img, target_color_space

def convert_from_ycrcb(img, dcspace):
    ''' convert image from YCrCb color space into destination color space'''
    target_color_space = dcspace
    if dcspace == 'HSV':
        converter = cv2.COLOR_YCrCb2HSV
    elif dcspace == 'LUV':
        converter = cv2.COLOR_YCrCb2LUV
    elif dcspace == 'HLS':
        converter = cv2.COLOR_YCrCb2HLS
    elif dcspace == 'YUV':
        converter = cv2.COLOR_YCrCb2YUV
    elif dcspace == 'YCrCb':
        converter = None
    elif dcspace == 'RGB':
        converter = cv2.COLOR_YCrCb2RGB
    elif dcspace == 'GRAY':
        converter = cv2.COLOR_YCrCb2GRAY
    elif dcspace == 'BGR':
        converter = cv2.COLOR_YCrCb2BGR
    else:
        print("Wrong dst color space:", dcspace)
        converter = None
        target_color_space = 'YCrCb'

    if converter != None:
        return cv2.cvtColor(img, converter), target_color_space
    else:
        return img, target_color_space

def convert_from_luv(img, dcspace):
    ''' convert image from LUV color space into destination color space'''
    target_color_space = dcspace
    if dcspace == 'HSV':
        converter = cv2.COLOR_LUV2HSV
    elif dcspace == 'LUV':
        converter = None
    elif dcspace == 'HLS':
        converter = cv2.COLOR_LUV2HLS
    elif dcspace == 'YUV':
        converter = cv2.COLOR_LUV2YUV
    elif dcspace == 'YCrCb':
        converter = cv2.COLOR_LUV2YCrCb
    elif dcspace == 'RGB':
        converter = cv2.COLOR_LUV2RGB
    elif dcspace == 'GRAY':
        converter = cv2.COLOR_LUV2GRAY
    elif dcspace == 'BGR':
        converter = cv2.COLOR_LUV2BGR
    else:
        print("Wrong dst color space:", dcspace)
        converter = None
        target_color_space = 'LUV'

    if converter != None:
        return cv2.cvtColor(img, converter), target_color_space
    else:
        return img, target_color_space

def color_convert(img, scspace, dcspace):
    '''do color convert accroding to input source color string and target color string'''
    target_color_space = dcspace
    if scspace == 'RGB':
        return convert_from_rgb(img, dcspace)
    elif scspace == 'BGR':
        return convert_from_bgr(img, dcspace)
    elif scspace == 'HSV':
        return convert_from_hsv(img, dcspace)
    elif scspace == 'HLS':
        return convert_from_hls(img, dcspace)
    elif scspace == 'YUV':
        return convert_from_yuv(img, dcspace)
    elif scspace == 'YCrCb':
        return convert_from_ycrcb(img, dcspace)
    elif scspace == 'LUV':
        return convert_from_luv(img, dcspace)
    else:
        print("input color space not supported:", scspace)
        return img, scspace

def color_convert_nocheck(img, scspace, dcspace, VERBOSE=False):
    '''do color convert without return result color space'''
    image, cspace = color_convert(img, scspace, dcspace)

    if VERBOSE:
        print("convert image from", scspace, "to",
              dcspace, "actual result is", cspace)
    return image


def get_cmap(cspace):
    '''find cmap according to color space string'''
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
