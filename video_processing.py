import os
import cv2
import sys
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from hog_subsample import find_cars
from multiple_detection import add_heat
from multiple_detection import apply_threshold
from multiple_detection import draw_labeled_bboxes
from scipy.ndimage.measurements import label
from vd_utils import load_classifier
from multiple_detection import draw_boxes
from color_convert import color_convert_nocheck


class CarDetector():
    def __init__(self, debug=False):
        self.num_avg_window = 3  # 3 frames sliding window
        self.heat = []
        self.num_processed_frames = 0
        self.num_history = 0
        self.avg_heat = []
        self.acc_heat = []
        self.debug = debug

    def update(self, heatmap):
        self.heat.append(np.copy(heatmap))
        # if self.num_history == 0:
        #    self.acc_heat = np.copy(heatmap)
        # else:
        #    self.acc_heat = self.acc_heat + heatmap
        self.num_history += 1
        # print(self.heat.shape)
        if (self.num_history > self.num_avg_window):
            old_heat = self.heat.pop(0)
        #    self.acc_heat -= old_heat
        #    self.avg_heat = np.copy(self.acc_heat) # / self.num_avg_window
        # else:
        #    self.avg_heat = np.copy(heatmap)
        if self.num_processed_frames > 0:
            #self.avg_heat = np.mean(self.heat, axis=0)
            self.avg_heat = np.stack(self.heat, axis=-1)

            self.avg_heat = np.mean(self.avg_heat, axis=2)
            '''
            h, w = self.heat[0].shape
            for i in range(0, h):
                for j in range(0, w):
                    found = False
                    for k in range(0, len(self.heat)):
                        if self.heat[k][i][j] != 0:
                            #print('[',k,'][',i,'][',j,']=', self.heat[k][i][j])
                            found = True
                    if found == True:
                        for k in range(0, len(self.heat)):
                            print('[',k,'][',i,'][',j,']=', self.heat[k][i][j])
                        print('heatmap [',i,'][',j,']=', heatmap[i][j])
                        print('avg [',i,'][',j,']=', self.avg_heat[i][j])
            '''
        else:
            self.avg_heat = np.squeeze(np.copy(self.heat))
        self.num_processed_frames += 1


def process_frame(img, svc, scaler, dcspace, spatial_size,
                  hist_bins, orient, pix_per_cell, cell_per_block,
                  hog_channel, spatial_feat, hist_feat, hog_feat):

    global detector
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    ystart = 400
    ystop = 656
    scspace = 'RGB'

    if detector.debug:
        window_img = color_convert_nocheck(img, scspace, 'BGR')

    #box_list = []
    for scale in (1.0, 1.5, 2.0):
        boxes = find_cars(img, scspace, dcspace, ystart, ystop, scale, svc, scaler,
                          orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        heat = add_heat(heat, boxes)
        if detector.debug:
            window_img = draw_boxes(window_img, boxes, thick=3)

    if detector.debug:
        savename = "output_images/{0:05d}.jpg".format(
            detector.num_processed_frames)
        cv2.imwrite(savename, window_img)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    detector.update(heatmap)
    #heatmap = np.clip(heat, 0, 255)
    # detector.update(heat)
    #heatmap = np.clip(detector.avg_heat, 0, 255)
    # if detector.debug:
    #    savename = "output_images/{0:05d}.jpg".format(detector.num_processed_frames)
    #    cv2.imwrite(savename, detector.acc_heat)
    heatmap = detector.avg_heat
    heatmap = apply_threshold(heatmap, 1)
    labels = label(heatmap)
    img = draw_labeled_bboxes(img, labels)

    return img


detector = CarDetector(debug=False)


def process_video(name):
    if os.path.exists(name) == False:
        print("Can't find file:", name)
        return
    basename = os.path.basename(name)
    head, ext = os.path.splitext(basename)
    output_video = head + '_output' + ext

    svc, scaler, dcspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = load_classifier()

    clip1 = VideoFileClip(name)
    output_clip = clip1.fl_image(lambda frame: process_frame(frame, svc, scaler, dcspace, spatial_size,
                                                             hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat))
    output_clip.write_videofile(output_video, audio=False)


if __name__ == "__main__":
    default = 'test_video.mp4'
    if len(sys.argv) == 1:
        print("use default video:", default)
    else:
        default = sys.argv.pop()
    process_video(default)
