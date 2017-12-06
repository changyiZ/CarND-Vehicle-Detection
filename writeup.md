**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_non_car.png
[image2]: ./output_images/car_rectangles.png
[image3]: ./output_images/detected_rectangles.png
[image4]: ./output_images/heatmap1.png
[image5]: ./output_images/heatmap2.png
[image6]: ./output_images/hog.png
[image7]: ./output_images/label_rectangles.png
[image8]: ./output_images/labeled_heatmap.png
[image9]: ./output_images/searching_areas.png
[image10]: ./output_images/sliding_window_searching.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
All of the code for the project is contained in the Jupyter notebook project.ipynb.

### Data Exploration

Since parts of the project vehicles dataset, the GTI* folders contain time-series data, simply ramdom train-test split would not
avoid overfitting.
So I divide it as follows: the first 80% of any folder containing images was assigned to be the training set, and the last 20% for the test set. Then the training set will be shuffled during the features extraction.
As showed in the 2nd code cell of *project.ipynb*.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The figure below shows a random sample of images from both classes of the dataset after loading all of the vehicle and non-vehicle image paths from the provided dataset. 

![alt text][image1]

The code for extracting HOG features from an image is defined by the method *get_hog_features*, titled "Define Method to Convert Image to Histogram of Oriented Gradients (HOG)." The figure below shows a comparison of a car image and its associated HOG, as well as the same for a non-car image.

![alt text][image6]

The method `extract_features` accepts a list of image paths and produces a flattened array of HOG features based on a variety of destination color spaces.
By using this method, features are extracted from the whole cars and non-car sample images.
These feature sets are combined and a label vector is defined (1 for cars, 0 for non-cars). The features and labels are then shuffled individually in training and test sets in preparation to be fed to a linear support vector machine (SVM) classifier. 
As describled in the section labeled *Make datasets ready for training* and *Train a Classifier*.


#### 2. Explain how you settled on your final choice of HOG parameters.
I experimented with a number of different combinations of color spaces and HOG parameters and trained a linear SVM using different combinations of HOG features extracted from the color channels. I discarded RGB color space, for its undesirable properties under changing light conditions. 
I considered not only the accuracy of which the classifier made predictions on the test dataset, but also the speed at which the classifier is able to make predictions. The strategy was to bias toward accuracy first, then achieve as fast of the training and extract features speed as possible.
The final parameters chosen are: *YUV* colorspace, 11 orientations, 16 pixels per cell, 2 cells per block, and `ALL` channels of the colorspace.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the default classifier parameters and using HOG features alone (I did not use spatial intensity or channel intensity histogram features) and was able to achieve a test accuracy above 98%.
Codes in the section labeled *Train a Classifier*.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled *Define method for using classifier to detect cars in an image* , the method `find_cars` combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive car prediction.

The image below shows the first attempt at using find_cars on one of the test images, using a single window size:

![alt text][image2]

I segmented the image into 4 partially overlapping zones with different sliding window sizes to account for different distances.
The following image show the configurations of all search windows in the final implementation, for small (1x), medium (1.5x, 2x), and large (3x) windows:

![alt text][image9]

A smaller (0.5) scaler in the previous implementations were found to return too many false positives, and originally the window overlap was set to 50% in both X and Y directions, but an overlap of 75% in the Y direction (yet still 50% in the X direction) produced more redundant true positive detections, which were preferable given the heatmap strategy. Additionally, only an appropriate vertical range of the image is considered for each window size (e.g. smaller range for smaller scales) to reduce the chance for false positives in areas where cars at that scale are unlikely to appear. The final implementation considers 190 window locations, which proved to be robust enough to detect vehicles reliably while maintaining a high speed of execution.

The image below shows the rectangles returned by find_cars drawn onto one of the test images in the final implementation.

![alt text][image10]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The results of passing all of the project test images through the above pipeline are displayed in the images below:

![alt text][image3]

The early implementation did not perform well, so I began by optimizing the SVM classifier. The original classifier used HOG features from the YUV Y channel only, and achieved a test accuracy of 96.28%. Using all three YUV channels increased the accuracy to 98.40%, but also tripled the execution time. However, changing the pixels_per_cell parameter from 8 to 16 produced a roughly ten-fold increase in execution speed with minimal cost to accuracy.
The final implementation performs very well, identifying the near-field vehicles in each of the images with no false positives.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are the corresponding heatmap of the test image:

![alt text][image4]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the frame:
![alt text][image8]

Here the resulting bounding boxes are drawn onto the frame:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

According to the result video, the detected bounding boxes sometimes may not cover the vehicle perfectly. It seems like the collision volume is not big enough to be safety.<br>
I think decreasing the threshold of heatmap would be helpful for that, but meanwhile it would introduce more false positives.
Another way to improve the heatmap is to scan the image more quickly.<br>
The pipeline may also failed for vehicles that significantly change position from one frame to the next, also dues to relatively slow calculation performance.<br>
The evaluation of feature vectors is currently done sequentially, but could be parallelized.<br>
Maybe some deep learning calculation based on GPU would be better, like the solution provided by [YOLO](https://pjreddie.com/darknet/yolo/).


