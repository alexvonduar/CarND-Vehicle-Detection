import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
from load_data import load_training_data


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features

def color_hist_feature(img, nbins=32, bins_range=(0, 256)):
    rhist, ghist, bhist, bin_centers, hist_features = color_hist(img=img, nbins=nbins, bins_range=bins_range)
    return hist_features

def test_color_hist(path):
    cars, noncars, data_info = load_training_data(path)
    ind = np.random.randint(0, len(cars))
    name = cars[ind]

    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rh, gh, bh, bincen, feature_vec = color_hist(
        image, nbins=32, bins_range=(0, 256))

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12, 4))
        #plt.suptitle(name, fontsize=16)
        plt.subplot(141)
        plt.imshow(image)
        plt.xlabel(name)
        plt.subplot(142)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(143)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(144)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
        fig.savefig("output_images/color_histogram.jpg")
    else:
        print('Your function is returning None for at least one variable...')


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    test_color_hist(test_dir)
