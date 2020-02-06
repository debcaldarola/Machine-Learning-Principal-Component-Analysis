# Machine Learning: Principal Components Analysis

<b>Principal Components Analysis</b> (PCA) is an unsupervised method that aims to find the smallest
subspace such that as much information about the original data as possible is preserved.<br>
Given a fixed number of points, PCA has as its goal to find an orthogonal set of linear basis vectors
such that the average reconstruction error is minimized. In the found optimal low-dimensional encoding
of the data the mean squared distance between the data point and their projection is minimized and
the variance of projected data maximized (the higher the value of the variance, the more information is
stored).<br>
Each Principal Component is orthogonal to the previous one and points in the direction of the largest
variance.<br>
PCA finds its main applications in data visualization, data compression and noise reduction.
The purpose of this experience is to apply such method in order to show what happens if different
principal components (PC) are chosen as basis for images representation and classification. Moreover, a
classifier has to be picked and applied to classify the images under different PC re-projections.<br>
In the following paper, the results will be expressed and analyzed.
