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
from load_data import load_training_data

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

def main(path, cspace = 'HSV'):
    cars, noncars, data_info = load_training_data(path)

    spatial = 32
    histbin = 32

    car_features = extract_color_features(cars, cspace=cspace, spatial_size=(spatial, spatial),
                                    hist_bins=histbin, hist_range=(0, 256))
    notcar_features = extract_color_features(noncars, cspace=cspace, spatial_size=(spatial, spatial),
                                       hist_bins=histbin, hist_range=(0, 256))

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    scaled_X = scale_norm(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:', spatial,
          'and', histbin, 'histogram bins')

    svc = color_svc(X_train, y_train)

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
    cspace = 'HSV'
    if len(sys.argv) == 1:
        print("use default dir:", test_dir, "color space", cspace)
    else:
        test_dir = sys.argv.pop()

    main(test_dir, cspace)