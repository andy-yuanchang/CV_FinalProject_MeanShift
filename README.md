# CV-Final-Project-MeanShift
This is a final project in a course called computer vision.
# Usage
The programming model inclues a function MeanShift(const IplImage*, int**). It will return a integer, the clustering result. The amount of colors depends on clustering number.

We can control two parameters, spatial_radius and color_radius. The input is a picture and output is a picture after doing meanshift.
# Program step
1. Convert the color space from RGB to L*U*V
