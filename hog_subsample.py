import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import sys

from color_convert import color_convert_nocheck
from hog_feature import get_hog_features
from bin_spatial import bin_spatial
from color_histogram import color_hist_feature
from vd_utils import load_classifier
from load_data import load_images
from load_data import load_image

# Define a single function that can extract features using hog
# sub-sampling and make predictions



def find_cars(img, scspace, dcspace, ystart, ystop, scale,
              svc, scaler, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, draw=False):

    if draw:
        #draw_img = np.copy(img)
        draw_img = color_convert_nocheck(img, scspace, 'RGB')
    #img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = color_convert_nocheck(img_tosearch, scspace, dcspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(
            imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)

    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist_feature(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                boxes.append(((xbox_left, ytop_draw + ystart),
                             (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                if draw:
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    if draw:
        return draw_img, boxes
    else:
        return boxes


def test_hog_scale(path):
    svc, scaler, dcspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = load_classifier()

    #input_name = 'test1.jpg'
    #img = mpimg.imread(input_name)

    ystart = 400
    ystop = 656
    scale = 1.5
    scspace = 'BGR'

    fnames = load_images(path)
    for fname in fnames:
        img = load_image(fname, scspace)
        print("processing", fname)
        for scale in (1.0, 1.5, 2.0):
            out_img, boxes = find_cars(img, scspace, dcspace, ystart, ystop, scale, svc, scaler,
                                orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, draw=True)

            # plt.imshow(out_img)

            basename = os.path.basename(fname)
            name, ext = os.path.splitext(basename)
            savename = os.path.join(
                'output_images', name + "_subsample_" + str(scale) + ext)
            fig = plt.figure()
            plt.imshow(out_img)
            fig.savefig(savename)
            plt.close()


if __name__ == "__main__":
    test_dir = "test_images"
    if len(sys.argv) == 1:
        print("use default test dir:", test_dir)
    else:
        test_dir = sys.argv.pop()

    test_hog_scale(test_dir)
