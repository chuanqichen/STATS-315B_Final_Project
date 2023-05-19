# STATS-315B Final Project

## Eigenfaces
The goal of this task is to learn how to recognize faces. We have a set of pictures of 20 people in various directions and expressions, some of which have sunglasses. One major problem with image data is that our input features are individual pixels, which are high-dimensional but not terribly meaningful in isolation. Using PCA, we can decompose our images into eigenvectors, which are linear combinations of pixels (nicknamed “eigenfaces”). Students can explore different classification tasks, from determining the presence of sunglasses to identifying individuals.

## Dataset
Data: Faces Directory   
Summary: This data consists of 640 black and white face images of people taken with varying pose (straight, left, right, up), expression (neutral, happy, sad, angry), eyes (wearing sunglasses or not), and size.     
Background: .PGM format specification  
http://netpbm.sourceforge.net/doc/pgm.html
```
wget --recursive --no-parent http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces/
find . -type f -name 'index.html*' -delete
```
```
Data Set Information:

Each image can be characterized by the pose, expression, eyes, and size. There are 32 images for each person capturing every combination of features.
To view the images, you can use the program xv.
The image data can be found in /faces. This directory contains 20 subdirectories, one for each person, named by userid. Each of these directories contains several different face images of the same person.
You will be interested in the images with the following naming convention:
.pgm
is the user id of the person in the image, and this field has 20 values: 
an2i, at33, boland, bpm, ch4f, cheyer, choon, danieln, glickman, karyadi, kawamura, 
kk49, megak, mitchell, night, phoebe, saavik, steffi, sz24, and tammo.
is the head position of the person, and this field has 4 values: straight, left, right, up.
is the facial expression of the person, and this field has 4 values: neutral, happy, sad, angry.
is the eye state of the person, and this field has 2 values: open, sunglasses.
is the scale of the image, and this field has 3 values: 1, 2, and 4. 
1 indicates a full-resolution image (128 columns by 120 rows); 
2 indicates a half-resolution image (64 by 60); 
4 indicates a quarter-resolution image (32 by 30).
If you've been looking closely in the image directories, you may notice that some images have a .bad suffix rather than the .pgm suffix. 
As it turns out, 16 of the 640 images taken have glitches due to problems with the camera setup; these are the .bad images. 
Some people had more glitches than others, but everyone who got ``faced'' should have at least 28 good face images 
(out of the 32 variations possible, discounting scale).
```

## Methods
dimensionality reduction, PCA, SVM, neural networks

## Timeline
- A project proposal (feedback, but no grade), due on May 1 (Mon) by 11:59pm.
- A project milestone (10% of the final grade), due on May 22 (Mon) by 11:59pm.
- A project poster presentation (10% of the final grade), on June 8 (Thu) from 10am-12pm at Sequoia coutyard.
- A final report (20% of the final grade), due on June 12 (Mon) by 11:59am (yes, am).

[Final Project](https://stanford-stats315b.github.io/spring2023/projects/)
