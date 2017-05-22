import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

import sys

from color_feature import extract_color_features
from color_feature import scale_norm
from load_data import load_training_images


def color_svc(X, y):
    print('Feature vector length:', len(X[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    return svc


def try_color_parameters(path):
    cars, noncars = load_training_images(path)

    sample_size = 2000
    cars = cars[0:sample_size]
    notcars = noncars[0:sample_size]
    print('select', sample_size, 'images from', len(cars),
          'car images and', len(noncars), 'non car images each')

    for spatial_color in 'RGB', 'YUV', 'LUV', 'YCrCb', 'GRAY', 'HLS', 'HSV':
        for hist_color in 'RGB', 'YUV', 'LUV', 'YCrCb', 'GRAY', 'HLS', 'HSV':
            for spatial_size in (16, 32):
                for histbin in (16, 32, 64):
                    car_features = extract_color_features(cars, spatial_color=spatial_color, spatial_size=(
                        spatial_size, spatial_size), hist_color=hist_color, hist_bins=histbin, hist_range=(0, 256))
                    notcar_features = extract_color_features(noncars, spatial_color=spatial_color, spatial_size=(
                        spatial_size, spatial_size), hist_color=hist_color, hist_bins=histbin, hist_range=(0, 256))

                    # Create an array stack of feature vectors
                    X = np.vstack((car_features, notcar_features)).astype(np.float64)
                    scaled_X, scaler = scale_norm(X)

                    # Define the labels vector
                    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

                    # Split up data into randomized training and test sets
                    rand_state = np.random.randint(0, 100)
                    X_train, X_test, y_train, y_test = train_test_split(
                        scaled_X, y, test_size=0.2, random_state=rand_state)

                    print('Using spatial color space of:', spatial_color, 'spatial binning of:', spatial_size,
                          '\nhistogram in color space of:', hist_color, 'and',  histbin, 'histogram bins')

                    svc = color_svc(X_train, y_train)

                    # Check the score of the SVC
                    print('Test Accuracy of SVC = ', round(
                        svc.score(X_test, y_test), 4))
                    # Check the prediction time for a single sample
                    t = time.time()
                    n_predict = 10
                    print('My SVC predicts: ',
                          svc.predict(X_test[0:n_predict]))
                    print('For these', n_predict,
                          'labels: ', y_test[0:n_predict])
                    t2 = time.time()
                    print(round(t2 - t, 5), 'Seconds to predict',
                          n_predict, 'labels with SVC')


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    try_color_parameters(test_dir)
