import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import sys
import os

from bin_spatial import bin_spatial
from color_histogram import color_hist_feature
from load_data import load_training_images
from load_data import load_image
from color_convert import color_convert_nocheck
from vd_utils import scale_norm


def extract_color_feature(fname, spatial_color, spatial_size, hist_color, hist_bins=32, hist_range=(0, 256)):
    ''' extract color features '''
    image = load_image(fname)

    spatial_image = color_convert_nocheck(image, 'BGR', spatial_color)

    spatial_feature = bin_spatial(spatial_image, size=spatial_size)

    hist_image = color_convert_nocheck(image, 'BGR', hist_color)
    hist_feature = color_hist_feature(
        hist_image, nbins=hist_bins, bins_range=hist_range)

    return np.concatenate((spatial_feature, hist_feature))


def extract_color_features(fnames, spatial_color, spatial_size, hist_color, hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for fname in fnames:
        feature = extract_color_feature(
            fname, spatial_color, spatial_size, hist_color, hist_bins=hist_bins, hist_range=hist_range)
        features.append(feature)
    # Return list of feature vectors
    return features


def main(path):
    cars, noncars = load_training_images(path)
    car_ind = np.random.randint(0, len(cars))
    noncar_ind = np.random.randint(0, len(noncars))

    for spatial_color in ('RGB', 'YUV', 'LUV', 'YCrCb', 'HLS', 'HSV', 'GRAY'):
        for hist_color in ('RGB', 'YUV', 'LUV', 'YCrCb', 'HLS', 'HSV', 'GRAY'):
            car_features = extract_color_features(cars, spatial_color=spatial_color, spatial_size=(
                32, 32), hist_color=hist_color, hist_bins=32, hist_range=(0, 256))
            notcar_features = extract_color_features(noncars, spatial_color=spatial_color, spatial_size=(
                32, 32), hist_color=hist_color, hist_bins=32, hist_range=(0, 256))

            if len(car_features) > 0:
                # Create an array stack of feature vectors
                X = np.vstack((car_features, notcar_features)
                              ).astype(np.float64)
                scaled_X, scaler = scale_norm(X)

                # Plot an example of raw and scaled features
                fig = plt.figure(figsize=(12, 8))

                plt.subplot(231)
                plt.imshow(load_image(cars[car_ind], 'RGB'))
                plt.title('Original Image')
                plt.xlabel(cars[car_ind])

                plt.subplot(232)
                plt.plot(X[car_ind])
                plt.title('Raw Features')

                plt.subplot(233)
                plt.plot(scaled_X[car_ind])
                plt.title('Normalized Features')

                plt.subplot(234)
                plt.imshow(load_image(noncars[noncar_ind], 'RGB'))
                plt.title('Original Image')
                plt.xlabel(noncars[noncar_ind])

                plt.subplot(235)
                plt.plot(X[noncar_ind + len(cars)])
                plt.title('Raw Features')

                plt.subplot(236)
                plt.plot(scaled_X[noncar_ind + len(cars)])
                plt.title('Normalized Features')

                plt.suptitle('spatial color: ' + spatial_color +
                             ' histogram color: ' + hist_color)
                fig.tight_layout()
                savename = os.path.join(
                    'output_images', spatial_color + '_' + hist_color + '_feature.jpg')
                fig.savefig(savename)
                plt.close()
            else:
                print('Your function only returns empty feature vectors...')


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    main(test_dir)
