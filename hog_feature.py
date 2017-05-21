import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sys
from skimage.feature import hog
from load_data import load_training_data

# Define a function to return HOG features and visualization


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(
            cell_per_block, cell_per_block), visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(
            cell_per_block, cell_per_block), visualise=False, feature_vector=feature_vec)
        return features


# Read in our vehicles and non-vehicles

def main(path):
    cars, noncars, data_info = load_training_data(path)

    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = cv2.imread(cars[ind])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False)

    fig = plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.imshow(image)
    plt.suptitle(cars[ind])
    plt.title('Example Car Image')
    plt.subplot(132)
    plt.imshow(gray, cmap='gray')
    plt.title('Gray Car Image')
    plt.subplot(133)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    fig.tight_layout()
    fig.savefig("output_images/hog.jpg")


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    main(test_dir)
