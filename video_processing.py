import os
import sys
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from hog_subsample import find_cars
from multiple_detection import add_heat
from multiple_detection import apply_threshold
from multiple_detection import draw_labeled_bboxes
from scipy.ndimage.measurements import label
from vd_utils import load_classifier


def process_frame(img):

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    ystart = 400
    ystop = 656
    scspace = 'RGB'

    svc, scaler, dcspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = load_classifier()

    #box_list = []
    for scale in (1.0, 1.5, 2.0):
        boxes = find_cars(img, scspace, dcspace, ystart, ystop, scale, svc, scaler,
                          orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        heat = add_heat(heat, boxes)

    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    img = draw_labeled_bboxes(img, labels)
    return img


def process_video(name):
    if os.path.exists(name) == False:
        print("Can't find file:", name)
        return
    basename = os.path.basename(name)
    head, ext = os.path.splitext(basename)
    output_video = head + '_processed' + ext

    clip1 = VideoFileClip(name)
    output_clip = clip1.fl_image(process_frame)
    output_clip.write_videofile(output_video, audio=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("please specify input video files")
    else:
        process_video(sys.argv.pop())
