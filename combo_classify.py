import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import sys
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from color_convert import color_convert_nocheck
from bin_spatial import bin_spatial
from color_histogram import color_hist_feature
from hog_feature import get_hog_features
from vd_utils import scale_norm
from load_data import load_training_images
from load_data import load_images
from vd_utils import load_classifier
from vd_utils import save_classifier
from vd_utils import have_classifier

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images


def combo_feature(img, scspace='BGR', dcspace='RGB', spatial_size=(32, 32),
                  hist_bins=32, orient=9,
                  pix_per_cell=8, cell_per_block=2, hog_channel='012',
                  spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    cvtImage = color_convert_nocheck(img, scspace, dcspace)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(cvtImage, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist_feature(cvtImage, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        hog_features = []
        if len(cvtImage.shape) == 3:
            channels = cvtImage.shape[2]
        else:
            channels = 1
        for channel in range(channels):
            if str(channel) in hog_channel:
                #print('use channel', channel)
                if channels == 1:
                    hog_features.append(get_hog_features(cvtImage[:, :],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
                else:
                    hog_features.append(get_hog_features(cvtImage[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_combo_features(fnames, dcspace='RGB', spatial_size=(32, 32),
                           hist_bins=32, orient=9,
                           pix_per_cell=8, cell_per_block=2, hog_channel=0,
                           spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for fname in fnames:
        # Read in each one by one
        image = cv2.imread(fname)
        # apply color conversion if other than 'RGB'
        feature = combo_feature(image, scspace='BGR', dcspace=dcspace, spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        features.append(feature)
    # Return list of feature vectors
    return features


def do_combo_feature_train(cars, notcars, dcspace='RGB', spatial_size=(32, 32),
                           hist_bins=32, orient=9,
                           pix_per_cell=8, cell_per_block=2, hog_channel='012',
                           spatial_feat=True, hist_feat=True, hog_feat=True, SAVE=True):
    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    #sample_size = 500
    #cars = cars[0:sample_size]
    #notcars = notcars[0:sample_size]
    print(len(cars), "cars ", len(notcars), "not cars")

    car_features = extract_combo_features(cars, dcspace=dcspace,
                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                          orient=orient, pix_per_cell=pix_per_cell,
                                          cell_per_block=cell_per_block,
                                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                                          hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_combo_features(notcars, dcspace=dcspace,
                                             spatial_size=spatial_size, hist_bins=hist_bins,
                                             orient=orient, pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block,
                                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                                             hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    scaled_X, scaler = scale_norm(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    if SAVE:
        save_classifier(svc, scaler, dcspace, spatial_size, hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
    return svc, scaler


def combo_feature_train(path, scspace='BGR', dcspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel='012',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    cars, noncars = load_training_images(path)

    svc, scaler = do_combo_feature_train(cars, noncars, dcspace=dcspace,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)
    return svc, scaler


def test_combo_feature(train_dir, test_dir, force_run = False):
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255
    scspace = 'BGR'
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

    if have_classifier() and force_run == False:
        svc, scaler, dcspace, spatial_size, hist_bins, orient, \
        pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = load_classifier()
    else:
        svc, scaler = combo_feature_train(train_dir, scspace=scspace, dcspace=dcspace,
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

        image = cv2.resize(image, (64, 64))

        features = combo_feature(image, scspace=scspace, dcspace=dcspace,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = svc.predict(test_features)

        basename = os.path.basename(fname)
        name, ext = os.path.splitext(basename)
        savename = os.path.join('output_images', name + "_classify" + ext)
        fig = plt.figure()
        plt.imshow(draw_image)
        if prediction == 1:
            plt.title('Original Image is car')
        else:
            plt.title('Original image is not car')
        plt.xlabel('fname')
        fig.savefig(savename)


if __name__ == "__main__":
    train_dir = "train_images"
    test_dir = "test_images/mytest_cars"
    if len(sys.argv) == 1:
        print("use default train dir:", train_dir,
              " default test dir:", test_dir)
    else:
        train_dir = sys.argv.pop()
        test_dir = sys.argv.pop()

    test_combo_feature(train_dir, test_dir)
