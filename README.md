# CV_FinalProject_MeanShift 
This is a final project in a course called computer vision.
# Usage #
The programming model inclues a function <code>MeanShift(const IplImage*, int**)</code>. It will return a integer, the clustering result. The amount of colors depends on clustering number.

We can control two parameters, <code>spatial_radius</code> and <code>color_radius</code>. The input is a picture and output is a picture after doing meanshift.
# Method #
## Mean Shift ##
1. Convert the color space from RGB to LUV
2. For every pixel calculate the mean vector when the pixel(x,y)'s distance between center < <code>spatial bandwidth</code>, color(l,u,v) of the pixel and the center’s distance < <code>range bandwidth</code>.
3. Let the original pixel’s color equal to the final center’s color.
4. Eliminate spatial regions containing less than <code>M</code> pixels.
## Clustering ##
- Connect neighboring pixel if they belong to same mode(cluster) using stack.

# Results #
The number after output respectively resprent (spitial_radius, color_radius, M).
## Figure 1 ##
<p float="left">
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image1/image1.jpg"/>
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image1/image1_result/filtered(16%2C19).png"/>
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image1/image1_result/cluster(16%2C19).png"/>
</p>

## Figure 2 ##
<p float="left">
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image2/image2.jpg"/>
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image2/image2_result/filtered.png"/>
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image2/image2_result/cluster.png"/>
</p>

## Figure 3 ##
<p float="left">
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image3/image3.jpg"/>
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image3/image3_result/filtered.png"/>
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image3/image3_result/cluster.png"/>
  <img src="https://github.com/YuAnChang1993/CV-Final-Project-MeanShift/blob/master/image3/image3_result/edge_result.png"/>
</p>



