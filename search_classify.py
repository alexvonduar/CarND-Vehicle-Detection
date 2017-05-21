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

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from combo_classify import combo_feature_train
from combo_classify import combo_feature
from sliding_window import slide_window
from sliding_window import draw_boxes
from load_data import load_images
from color_convert import color_convert_nocheck
from vd_utils import have_classifier
from vd_utils import load_classifier

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())


def search_windows(img, windows, clf, scaler, scspace='BGR', dcspace='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel='012', spatial_feat=True,
                   hist_feat=True, hog_feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        # print(window)
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = combo_feature(test_img, scspace=scspace, dcspace=dcspace,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


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

        min_window = spatial_size[0]
        max_window = 256

        top_y = int(image.shape[0] / 2)
        y_start_stop = [top_y, image.shape[0]]
        valid_y = image.shape[0] - top_y
        for window_size in range(min_window, max_window, spatial_size[0]):
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

            window_img = draw_boxes(draw_image, hot_windows,
                                    color=(0, 0, 255), thick=3)

            plt.imshow(window_img)
            basename = os.path.basename(fname)
            name, ext = os.path.splitext(basename)
            savename = os.path.join(
                'output_images', name + "_" + str(window_size) + "_" + ext)
            fig = plt.figure()
            plt.imshow(window_img)
            plt.title('Original Image')
            plt.xlabel('fname')
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
