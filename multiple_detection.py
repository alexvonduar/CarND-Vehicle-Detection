import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import sys
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from combo_classify import combo_feature_train
from combo_classify import combo_feature
from sliding_window import slide_window
from sliding_window import draw_boxes
from load_data import load_images
from color_convert import color_convert_nocheck
from vd_utils import have_classifier
from vd_utils import load_classifier
from scipy.ndimage.measurements import label
from search_classify import search_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img

def get_labeled_bboxes(labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    # Return the image
    return bboxes

def draw_labeled_bboxes_solid(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        #cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] = 1
    # Return the image
    return img

def apply_window_search(train_dir, test_dir):
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255
    dcspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = '012'  # Can be 0, 1, 2, or "012"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [None, None]  # Min and max in y to search in slide_window()
    if have_classifier():
        svc, scaler, dcspace, spatial_size, hist_bins, orient, \
            pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = load_classifier()
    else:
        svc, scaler = combo_feature_train(train_dir, scspace='BGR', dcspace=dcspace,
                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                          orient=orient, pix_per_cell=pix_per_cell,
                                          cell_per_block=cell_per_block,
                                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                                          hist_feat=hist_feat, hog_feat=hog_feat)

    fnames = load_images(test_dir)

    for fname in fnames:
        image = cv2.imread(fname)
        draw_image = np.copy(image)
        draw_image = color_convert_nocheck(image, 'BGR', 'RGB')
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        min_window = spatial_size[0]
        max_window = 256

        top_y = int(image.shape[0] / 2)
        y_start_stop = [top_y, image.shape[0]]
        valid_y = image.shape[0] - top_y
        for window_size in range(min_window, max_window, min_window):
            #window_size = int(min_window * scale_factor)
            #print("window size", window_size, "larger than", image.shape[0], "x", image.shape[1])

            if window_size > valid_y or window_size > image.shape[1]:
                print("window size", window_size, "larger than",
                      valid_y, "x", image.shape[1])
                continue

            windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                   xy_window=(window_size, window_size), xy_overlap=(0.75, 0.75))

            hot_windows = search_windows(image, windows, svc, scaler, scspace='BGR', dcspace=dcspace,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)

            add_heat(heat, hot_windows)

            window_img = draw_boxes(draw_image, hot_windows,
                                    color=(0, 0, 255), thick=1)

            #plt.imshow(window_img)
            basename = os.path.basename(fname)
            name, ext = os.path.splitext(basename)
            savename = os.path.join(
                'output_images', name + "_" + str(window_size) + "_" + ext)
            fig = plt.figure()
            plt.imshow(window_img)
            plt.title('Searchin window size: ' + str(window_size))
            plt.xlabel(fname)
            fig.savefig(savename)
            plt.close()
        heat = apply_threshold(heat, 5)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(draw_image), labels)
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        basename = os.path.basename(fname)
        name, ext = os.path.splitext(basename)
        savename = os.path.join('output_images', name + "_heat" + ext)
        fig.savefig(savename)
        plt.close()


if __name__ == "__main__":
    train_dir = "train_images"
    test_dir = "test_images"
    if len(sys.argv) == 1:
        print("use default train dir:", train_dir,
              " default test dir:", test_dir)
    else:
        train_dir = sys.argv.pop()
        test_dir = sys.argv.pop()

    apply_window_search(train_dir, test_dir)
