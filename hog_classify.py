import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sys
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

from hog_feature import get_hog_features
from load_data import load_training_images
from load_data import load_image
#from color_convert import color_convert_nocheck
from vd_utils import scale_norm


def extract_hog_features(fnames, cspace='RGB', orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel='012'):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for fname in fnames:
        # apply color conversion if other than 'RGB'
        image = load_image(fname, cspace)

        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = []
        if len(image.shape) == 3:
            channels = image.shape[2]
        else:
            channels = 1
        for channel in range(channels):
            if channels == 1:
                hog_features.append(get_hog_features(
                    image[:, :], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            elif str(channel) in hog_channel:
                hog_features.append(get_hog_features(
                    image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
        if len(hog_features) == 0:
            break
        hog_features = np.ravel(hog_features)
        features.append(hog_features)
    return features


def hog_classifier(X, y):
    print('Feature vector length:', len(X[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    return svc


def test_hog_feature(path):
    cars, noncars = load_training_images(path)

    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    '''
    sample_size = 2000
    cars = cars[0:sample_size]
    notcars = noncars[0:sample_size]
    print('select 1000 images from', len(cars),
          'car images and', len(noncars), 'non car images each')
        '''

    # TODO: Tweak these parameters and see how the results change.
    colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = '012'  # Can be 0, 1, 2, or "ALL"

    for colorspace in ('GRAY'):#'RGB', 'YUV', 'HLS', 'HSV', 'LUV', 'YCrCb', 'GRAY'):
        colorspace = 'GRAY'
        for hog_channel in ('0'):#, '1', '2', '01', '02', '12', '012'):
            print('use color', colorspace, 'channel', hog_channel)

            t = time.time()
            car_features = extract_hog_features(cars, cspace=colorspace, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel)
            notcar_features = extract_hog_features(noncars, cspace=colorspace, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel)

            if len(car_features) == 0 or len(notcar_features) == 0:
                print("can't find", hog_channel, 'in', colorspace)
                break

            t2 = time.time()
            print(round(t2 - t, 2), 'Seconds to extract HOG features...')
            # Create an array stack of feature vectors
            X = np.vstack((car_features, notcar_features)).astype(np.float64)
            scaled_X, scaler = scale_norm(X)

            # Define the labels vector
            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)

            print('Using: color', colorspace, 'channel', hog_channel, orient, 'orientations', pix_per_cell,
                  'pixels per cell and', cell_per_block, 'cells per block')

            svc = hog_classifier(X_train, y_train)

            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t = time.time()
            n_predict = 10
            print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
            print('For these', n_predict, 'labels: ', y_test[0:n_predict])
            t2 = time.time()
            print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    test_hog_feature(test_dir)
