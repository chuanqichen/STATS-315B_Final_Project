# STATS-315B Final Project

## Eigenfaces
The goal of this task is to learn how to recognize faces. We have a set of pictures of 20 people in various directions and expressions, some of which have sunglasses. One major problem with image data is that our input features are individual pixels, which are high-dimensional but not terribly meaningful in isolation. Using PCA, we can decompose our images into eigenvectors, which are linear combinations of pixels (nicknamed “eigenfaces”). Students can explore different classification tasks, from determining the presence of sunglasses to identifying individuals.

## Dataset
Data: Faces Directory   
Background: .PGM format specification  
http://netpbm.sourceforge.net/doc/pgm.html
```
wget --recursive --no-parent http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces/
find . -type f -name 'index.html*' -delete
```

## Methods
dimensionality reduction, PCA, SVM, neural networks

## Timeline
- A project proposal (feedback, but no grade), due on May 1 (Mon) by 11:59pm.
- A project milestone (10% of the final grade), due on May 22 (Mon) by 11:59pm.
- A project poster presentation (10% of the final grade), on June 8 (Thu) from 10am-12pm at Sequoia coutyard.
- A final report (20% of the final grade), due on June 12 (Mon) by 11:59am (yes, am).

[Final Project](https://stanford-stats315b.github.io/spring2023/projects/)
