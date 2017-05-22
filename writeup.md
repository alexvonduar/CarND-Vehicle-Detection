**Vehicle Detection Project**
---

---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/training_data.jpg
[image2]: ./output_images/bin_spatial.jpg
[image3]: ./output_images/RGB_hist.jpg
[image4]: ./output_images/YCrCb_hist.jpg
[image5]: ./output_images/hog.jpg
[image6]: ./output_images/YCrCb_YCrCb_feature.jpg
[image7]: ./output_images/test5_window.jpg
[image8]: ./output_images/test5_64_.jpg
[image9]: ./output_images/test5_subsample_1.0.jpg
[image10]: ./output_images/test5_subsample_1.5.jpg
[image11]: ./output_images/test5_subsample_2.0.jpg
[image12]: ./output_images/test5_heatmap.jpg
[image13]: ./output_images/test5_labels.jpg
[image14]: ./output_images/00199_debug.jpg
[image24]: ./examples/sliding_window.jpg
[image25]: ./examples/bboxes_and_heat.png
[image26]: ./examples/labels_map.png
[image27]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---


## 1. Classifier Training
### 1. 1 Training Data

I download training sets of [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images from lesson as the training data. According to project tips, I apply random shuffle when travese the training image dirs to list all the input image names in function `load_training_images()` in file `load_data.py:36-54`. From the two image sets, I can see there are 8792 images of cars and 8968 images of non-cars. 

![alt text][image1]

Since the number of cars and not-cars images seems to be in balance, I decide to use them directly without any augment process.

### 1. 2 Features

Since I decied to use a SVM classifier, so I need to extract some features from traing images that can feed to classifier and train it to work.

From the lesson, I can have a spatial binning, color histogram and HOG to choose or some combination of them.

First, I try to extract spatial binning and visualize the result. To get that, I use function `bin_spatial()` in file `bin_spatial.py:12-14`.

![alt text][image2]

I then explored histogram of different color spaces, use function `color_hist_feature()` in file `color_histogram.py:12-48`. Here's an example of `RGB` and `YCrCb`:

![alt text][image3]
![alt text][image4]  

For HOG feature, there are different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I use function `get_hog_features()` in file `hog_feature.py:14-24` to calculate hog result, and function `test_hog()` in file `hog_feature.py:29-76` to grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image5]

### 1. 3 Classifier and Parameters

There are many parameters that can be tuned for extract features:
1. binning spatial: `spatial_size`
2. histogram: `hist_bins`, `hist_range`
3. HOG: `orientations`, `pixels_per_cell`, `cells_per_block`, `hog_channel`

I tried some combinations of parameters using functions in file `color_classify.py` and `hog_classify.py` to help to evaluate the performance of different parameter set. I didn't try every combination that possible. From the color classify test, I can see most combinations are almost at the same acuracy level except using gray color space. From the HOG test, I find YUV, LUV, YCrCb and HSV color space out performed than other color spaces a little bit, all around accuracy from ~99.8% to ~99.4%.

Finally, according to prior tests, I choose the `spatial_size` as 32, choose `hist_bins=32` and keep the default `hist_range=(0,256)`. I choose HOG parameters `orientations=9`, `pixels_per_cell=8`, `cells_per_block=2`, `hog_channel=012` which means do HOG process for each possible color channels. And I choose to use `YCrCb` color space. Here's a table for the final parameter set.

|Parameters||
|:----:|:----:|
|color space| YCrCb|
|spatial size| 32x32|
|histogram bins|32|
|histogram range|[0, 255]|
|hog orientations|9|
|hog pixels per cell|8|
|hog cells per block|2|
|hog channels| all|
|||

Here's an example of extracted features:

![alt text][image6]

Feature extraction is implemented by function `combo_feature()` in file `combo_classify.py:32-73`. In function `do_combo_feature_train()` of file `combo_classify.py:108-121`, I extract all the feature of cars and non-cars images and concatenate to one list, then standarized(`scale_norm()`, `vd_utils.py:6-11`) the list using `sklearn.StandardScaler()` so that all features are scaled to zero mean and unit variance before training the classifier.

Since I've aready shuffle the input data(`load_training_images()`, `load_data.py:52-53`), I just simply partitioned it into training(80%) and testing(20%) set(`do_combo_feature_train()`, `combo_classify.py:129-130`) using `sklearn.train_test_split()`.

Finally, I put them all into a linear SVM, `LinearSVC()`, in function `do_combo_feature_train()`, file `combo_classify.py:132-143`. Using prior parameter set, I got final test accuracy at 99.3%. After training, I save model as `svc.pkl` file for use later.

## 2. Sliding Window Search

Fist, I try and show a sliding window search in funcion `slide_window()` in file `sliding_window.py:34-72`. The ROI was set to height 400 to 656, 256 pixel lines of input images, and search window is 64 pixels, overlap set to 0.75. Here's an example, red square is the first search window, and search step is 1/4 searching window width.

![alt text][image7]

To improve the performance, beside ROI, I also adopt the HOG sub-sampleing window search. It is implement in function `find_cars()`, file `hog_subsample.py:22-106`. Set window size to 64, hog parameters to be the same as prior `(orientations=9, pixels_per_cell=8, cells_per_block=2)`. I just set `cells_per_step=2`, which means the same overlap 0.75 as sliding window implementation. While I choose scale factor to be 1.0, 1.5 and 2.0, so, actually equivalent search window are 64x64 for factor 1.0, 96x96 for 1.5, and 128x128 for 2.0.

Scale Factor 1.0

![alt text][image9]

Scale Factor 1.5

![alt text][image10]

Scale Factor 2.0

![alt text][image11]

After searching with different scale, I record boxes found in each scale, and add up all the boxes to form a head map, reference code are in function `test_hog_scale()` file `hog_subsample.py:145`:

![alt text][image12]

Then I apply threashold on it and use `scipy.ndimage.measurements.label()` to generate car bounding boxes, function `test_hog_scale()` file `hog_subsample.py:147`

![alt text][image13]

---

## 4. Video Implementation

All after all, to process video frames and minimize fake positive. I use a CarDetector class, file `video_processing.py:16-65`. I'll save car bounding boxes of previous 3 frames, and find car bounding boxes of current frame whether or not overlap with previous boxes, if not, mark it as fake positive, otherwise, mark it as valid box, function `filter_out()` in file `video_processing.py:44-52`.

Here's an example image, blue rectangle is the positive, yellow boxes are all the boxes finding before thresholding, and red one is marked the fake positive.

![alt text][image14] 


Apply video processing on test videos, here is the [result](./project_video_output.mp4)


---

## 5. Discussion

In this project, I use a linear SVM classifier with spatial, histogram and HOG features together to identify the cars, then I try to use spatial sliding window and to find the cars in images and video frames, and I also try temperal sliding window to mask out the fake positives. I think to find a suitable parameter set is really a challenge for SVM learning method, too many combinations to try. Maybe if use CNN or other NN methods could make this step more steady. From the result, the car boundaries are changing quickly from frame to frame, so, I think the temperal sliding window mechanism should not only pick the false positives, but also make smoothing between frames.