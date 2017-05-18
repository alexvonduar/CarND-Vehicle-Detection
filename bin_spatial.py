import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from load_data import load_training_data
from color_convert import color_convert
from color_convert import get_cmap


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, size=(32, 32)):
    img = cv2.resize(img, size).ravel()
    return img


# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples

def test_bin_spatial(path, cspace='RGB'):
    cars, noncars, data_info = load_training_data(path)
    ind = np.random.randint(0, len(cars))
    name = cars[ind]
    # Read in an image
    # You can also read cutout2, 3, 4 etc. to see other examples
    image = cv2.imread(name)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cvt = color_convert(image, 'BGR', cspace)

    feature_vec = bin_spatial(cvt, size=(32, 32))

    # Plot features
    fig = plt.figure(figsize=(12, 2))
    plt.subplot(121)
    plt.imshow(color_convert(image, 'BGR', 'RGB'))
    plt.xlabel(name)
    plt.subplot(122)
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')
    plt.suptitle(cspace)
    fig.savefig("output_images/bin_spatial.jpg")


if __name__ == "__main__":
    test_dir = "train_images"
    cspace = 'HLS'
    if len(sys.argv) == 1:
        print("use default dir:", test_dir, "color space", cspace)
    else:
        test_dir = sys.argv.pop()

    test_bin_spatial(test_dir, cspace)
