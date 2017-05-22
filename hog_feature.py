import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sys
from skimage.feature import hog
from load_data import load_training_images
from load_data import load_image

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

def test_hog(path):
    cars, noncars = load_training_images(path)

    # Generate a random index to look at a car image
    car_ind = np.random.randint(0, len(cars))
    noncar_ind = np.random.randint(0, len(noncars))
    # Read in the image
    car_image = load_image(cars[car_ind], 'RGB')
    car_gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
    noncar_image = load_image(noncars[noncar_ind], 'RGB')
    noncar_gray = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, car_hog_image = get_hog_features(car_gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False)
    features, noncar_hog_image = get_hog_features(noncar_gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False)


    fig = plt.figure(figsize=(10, 8))
    plt.subplot(231)
    plt.imshow(car_image)
    plt.xlabel(cars[car_ind])
    plt.title('Example Car Image')
    plt.subplot(232)
    plt.imshow(car_gray, cmap='gray')
    plt.title('Gray Car Image')
    plt.subplot(233)
    plt.imshow(car_hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.subplot(234)
    plt.imshow(noncar_image)
    plt.xlabel(noncars[noncar_ind])
    plt.title('Example Non-car Image')
    plt.subplot(235)
    plt.imshow(noncar_gray, cmap='gray')
    plt.title('Gray Non-car Image')
    plt.subplot(236)
    plt.imshow(noncar_hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.suptitle('HOG')
    fig.tight_layout()
    fig.savefig("output_images/hog.jpg")


if __name__ == "__main__":
    test_dir = "train_images"
    if len(sys.argv) == 1:
        print("use default dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    test_hog(test_dir)
