# CV-Final-Project-MeanShift
This is a final project in a course called computer vision.
# Usage
The programming model inclues a function MeanShift(const IplImage*, int**). It will return a integer, the clustering result. The amount of colors depends on clustering number.

We can control two parameters, spatial_radius and color_radius. The input is a picture and output is a picture after doing meanshift.
# Program step
## Mean Shift
1. Convert the color space from RGB to LUV

2. For every pixel calculate the mean vector when the pixel(x,y)'s distance between center < spatial bandwidth, color(l,u,v) of the pixel and the center’s distance < range bandwidth.

3. Let the original pixel’s color equal to the final center’s color.

4. Eliminate spatial regions containing less than M pixels.



