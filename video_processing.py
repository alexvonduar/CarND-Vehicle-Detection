import os
import cv2
import sys
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from hog_subsample import find_cars
from multiple_detection import add_heat
from multiple_detection import apply_threshold
from multiple_detection import get_labeled_bboxes
from scipy.ndimage.measurements import label
from vd_utils import load_classifier
from multiple_detection import draw_boxes
from color_convert import color_convert_nocheck

def is_valid_box(box, boxes):
    valid = False
    count = 0
    for i in range(len(boxes)):
        box_list = boxes[i]
        for j in range(len(box_list)):
            ref_box = box_list[j]
            (rx0, ry0) = ref_box[0]
            (rx1, ry1) = ref_box[1]
            (x0, y0) = box[0]
            (x1, y1) = box[1]
            if ((rx1 - x0) * (rx0 - x1) <= 0) and ((ry1 - y0) * (ry0 - y1) <= 0):
                count += 1
                break
    valid = count > (len(boxes) / 2)
    if valid == False:
        print('invalid box', box)
    return count > (len(boxes) / 2)

class CarDetector():
    def __init__(self, debug=False):
        self.num_avg_window = 3  # 3 frames sliding window
        self.boxes = []
        self.num_processed_frames = 0
        self.num_history = 0
        self.valid_boxes = []
        self.invalid_boxes = []
        self.debug = debug

    def filter_out(self, boxes):
        valid_boxes = []
        invalid_boxes = []
        for box in boxes:
            if is_valid_box(box, self.boxes):
                valid_boxes.append(box)
            else:
                invalid_boxes.append(box)
        return valid_boxes, invalid_boxes

    def update(self, boxes):
        if self.num_processed_frames > 0:
            self.valid_boxes, self.invalid_boxes = self.filter_out(boxes)
        else:
            self.valid_boxes = boxes
            self.invalid_boxes = []

        self.boxes.append(np.copy(boxes))
        self.num_history += 1
        if (self.num_history > self.num_avg_window):
            old_boxes = self.boxes.pop(0)
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
            window_img = draw_boxes(window_img, boxes, color=(0, 255, 255), thick=2)

    #if detector.debug:
        #savename = "output_images/{0:05d}.jpg".format(
        #    detector.num_processed_frames)
        #cv2.imwrite(savename, window_img)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    #detector.update(heatmap)
    #heatmap = np.clip(heat, 0, 255)
    # detector.update(heat)
    #heatmap = np.clip(detector.avg_heat, 0, 255)
    # if detector.debug:
    #    savename = "output_images/{0:05d}.jpg".format(detector.num_processed_frames)
    #    cv2.imwrite(savename, detector.acc_heat)
    #heatmap = detector.avg_heat
    #heatmap = apply_threshold(heatmap, 1)
    labels = label(heatmap)
    #img = draw_labeled_bboxes(img, labels)
    bboxes = get_labeled_bboxes(labels)
    detector.update(bboxes)
    img = draw_boxes(img, detector.valid_boxes)
    if detector.debug:
        window_img = draw_boxes(window_img, detector.valid_boxes, color=(255, 0, 0), thick=3)
        window_img = draw_boxes(window_img, detector.invalid_boxes, color=(0,0,255), thick=5)
        savename = "output_images/{0:05d}_debug.jpg".format(detector.num_processed_frames)
        cv2.imwrite(savename, window_img)

    return img


detector = CarDetector(debug=True)


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
