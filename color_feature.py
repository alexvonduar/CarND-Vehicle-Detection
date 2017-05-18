import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import sys

from bin_spatial import bin_spatial
from color_histogram import color_hist_feature
from load_data import load_training_data
from color_convert import color_convert
from vd_utils import scale_norm

def extract_color_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        cvt = color_convert(image, 'BGR', cspace)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(image, size=spatial_size)
        # Apply color_hist_features() also with a color space option now
        hist_features = color_hist_feature(
            cvt, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


def main(path, cspace='HLS'):
    cars, noncars, data_info = load_training_data(path)

    car_features = extract_color_features(cars, cspace=cspace, spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256))
    notcar_features = extract_color_features(noncars, cspace=cspace, spatial_size=(32, 32),
                                       hist_bins=32, hist_range=(0, 256))

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        scaled_X = scale_norm(X)
        car_ind = np.random.randint(0, len(cars))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.xlabel(cars[car_ind])
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.xlabel('')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        fig.savefig("output_images/color_feature.jpg")
    else:
        print('Your function only returns empty feature vectors...')

if __name__ == "__main__":
    test_dir = "train_images"
    cspace = 'HLS'
    if len(sys.argv) == 1:
        print("use default dir:", test_dir, "color space", cspace)
    else:
        test_dir = sys.argv.pop()

    main(test_dir, cspace)
