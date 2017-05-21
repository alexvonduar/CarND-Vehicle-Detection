import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
from load_data import load_training_images
from load_data import load_image
from color_convert import color_convert


def color_hist(img, nbins=32, bins_range=(0, 256)):
    ''' calculate histogram of all channels'''
    shape = img.shape
    if len(shape) == 2:
        num_channels = 1
    else:
        num_channels = img.shape[2]

    if num_channels == 1:
        hist0 = np.histogram(img[:, :], bins=nbins, range=bins_range)
    else:
        hist0 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        hist1 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)

        if num_channels > 2:
            hist2 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Generating bin centers
    bin_edges = hist0[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

    # Concatenate the histograms into a single feature vector
    if num_channels == 3:
        hist_features = np.concatenate((hist0[0], hist1[0], hist2[0]))
        return hist_features, bin_centers, num_channels, hist0, hist1, hist2
    elif num_channels == 1:
        hist_features = hist0[0]  # np.concatenate((hist0[0]))
        return hist_features, bin_centers, num_channels, hist0, None, None
    else:
        return None, None, 0, None, None, None
    # Return the individual histograms, bin_centers and feature vector


def color_hist_feature(img, nbins=32, bins_range=(0, 256)):
    hist_features, bin_centers, num_channels, hist0, hist1, hist2 = color_hist(
        img=img, nbins=nbins, bins_range=bins_range)
    return hist_features


def test_color_hist(path):
    cars, noncars = load_training_images(path)

    car_ind = np.random.randint(0, len(cars))
    noncar_ind = np.random.randint(0, len(noncars))
    car_name = cars[car_ind]
    noncar_name = noncars[noncar_ind]

    for cspace in ('RGB', 'YUV', 'HLS', 'LUV', 'YCrCb', 'HSV', 'GRAY'):
        car_image = load_image(car_name, cspace)
        noncar_image = load_image(noncar_name, cspace)

        car_feature_vec, car_bincen, car_num_channels, car_ch0, car_ch1, car_ch2 = color_hist(
            car_image, nbins=32, bins_range=(0, 256))
        noncar_feature_vec, noncar_bincen, noncar_num_channels, noncar_ch0, noncar_ch1, noncar_ch2 = color_hist(
            noncar_image, nbins=32, bins_range=(0, 256))

        # Plot a figure with all channel bars
        if car_num_channels == 1:
            fig = plt.figure(figsize=(12, 6))
            plt.subplot(231)
            plt.imshow(load_image(car_name, 'RGB'))
            plt.title(car_name)

            plt.subplot(232)
            plt.imshow(car_image, cmap='gray')
            plt.xlabel(cspace)

            plt.subplot(233)
            plt.bar(car_bincen, car_ch0[0])
            plt.xlim(0, 256)
            plt.title('Histogram')

            plt.subplot(234)
            plt.imshow(load_image(noncar_name, 'RGB'))
            plt.title(noncar_name)

            plt.subplot(235)
            plt.imshow(noncar_image, cmap='gray')
            plt.xlabel(cspace)

            plt.subplot(236)
            plt.bar(noncar_bincen, noncar_ch0[0])
            plt.xlim(0, 256)
            plt.title('Histogram')
            plt.suptitle(cspace)
            # fig.tight_layout()
            savename = os.path.join('output_images', cspace + '_hist.jpg')
            fig.savefig(savename)
        elif car_num_channels == 3:
            fig = plt.figure(figsize=(20, 6))
            plt.suptitle(cspace)

            plt.subplot(271)
            plt.imshow(load_image(car_name, 'RGB'))
            plt.title(car_name)

            plt.subplot(272)
            plt.imshow(car_image[:, :, 0], cmap='gray')
            plt.xlabel('Channel 1')

            plt.subplot(273)
            plt.imshow(car_image[:, :, 1], cmap='gray')
            plt.xlabel('Channel 2')

            plt.subplot(274)
            plt.imshow(car_image[:, :, 2], cmap='gray')
            plt.xlabel('Channel 3')

            plt.subplot(275)
            plt.bar(car_bincen, car_ch0[0])
            plt.xlim(0, 256)
            plt.xlabel('Channel 1 Histogram')

            plt.subplot(276)
            plt.bar(car_bincen, car_ch1[0])
            plt.xlim(0, 256)
            plt.xlabel('Channel 2 Histogram')

            plt.subplot(277)
            plt.bar(car_bincen, car_ch2[0])
            plt.xlim(0, 256)
            plt.xlabel('Channel 3 Histogram')

            plt.subplot(278)
            plt.imshow(load_image(noncar_name, 'RGB'))
            plt.title(noncar_name)

            plt.subplot(279)
            plt.imshow(noncar_image[:, :, 0], cmap='gray')
            plt.xlabel('Channel 1')

            plt.subplot(2, 7, 10)
            plt.imshow(noncar_image[:, :, 1], cmap='gray')
            plt.xlabel('Channel 2')

            plt.subplot(2, 7, 11)
            plt.imshow(noncar_image[:, :, 2], cmap='gray')
            plt.xlabel('Channel 3')

            plt.subplot(2, 7, 12)
            plt.bar(noncar_bincen, noncar_ch0[0])
            plt.xlim(0, 256)
            plt.xlabel('Channel 1 Histogram')

            plt.subplot(2, 7, 13)
            plt.bar(noncar_bincen, noncar_ch1[0])
            plt.xlim(0, 256)
            plt.xlabel('Channel 2 Histogram')

            plt.subplot(2, 7, 14)
            plt.bar(noncar_bincen, noncar_ch2[0])
            plt.xlim(0, 256)
            plt.xlabel('Channel 3 Histogram')

            # fig.tight_layout()
            savename = os.path.join('output_images', cspace + '_hist.jpg')
            fig.savefig(savename)
        else:
            print('Your function is returning None for at least one variable...')


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    test_color_hist(test_dir)
