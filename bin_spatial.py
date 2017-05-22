import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from load_data import load_training_images
from load_data import load_image
from color_convert import color_convert_nocheck
from color_convert import get_cmap


def bin_spatial(img, size=(32, 32)):
    img = cv2.resize(img, size).ravel()
    return img


def test_bin_spatial(path, dcspace='RGB'):
    cars, noncars = load_training_images(path)

    target_cspace = dcspace
    # read a car image and get feature
    ind = np.random.randint(0, len(cars))
    car_name = cars[ind]
    car_image = load_image(car_name, dcspace)
    car_feature = bin_spatial(car_image, size=(32, 32))

    # read a non car image and get feature
    ind = np.random.randint(0, len(noncars))
    noncar_name = noncars[ind]
    noncar_image = load_image(noncar_name, dcspace)
    noncar_feature = bin_spatial(noncar_image, size=(32, 32))

    # Plot features
    fig = plt.figure()
    plt.subplot(221)
    plt.imshow(color_convert_nocheck(car_image, dcspace, 'RGB'))
    plt.xlabel(car_name)
    plt.subplot(222)
    plt.plot(car_feature)
    plt.title('Spatially Binned Features')
    plt.suptitle(dcspace)
    plt.subplot(223)
    plt.imshow(color_convert_nocheck(noncar_image, dcspace, 'RGB'))
    plt.xlabel(noncar_name)
    plt.subplot(224)
    plt.plot(noncar_feature)
    plt.title('Spatially Binned Features')
    plt.suptitle(dcspace)
    fig.savefig("output_images/bin_spatial.jpg")
    plt.show()
    plt.close()


if __name__ == "__main__":
    test_dir = "train_images"
    dcspace = 'YCrCb'
    if len(sys.argv) == 1:
        print("use default dir:", test_dir, "color space", dcspace)
    else:
        dcspace = sys.argv.pop()
        test_dir = sys.argv.pop()

    test_bin_spatial(test_dir, dcspace)
