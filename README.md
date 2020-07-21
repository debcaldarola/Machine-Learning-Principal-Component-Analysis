# Machine Learning and Artificial Intelligence Course Assignments
Reports and coding for Machine Learning and Artificial Intelligence Course @ Politecnico di Torino

## HW1: Principal Components Analysis

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

## HW2: Support Vector Machines
<b>Support Vector Machines</b> (SVMs) are a set of supervised learning methods usually applied to detect
outliers, for regression and classification algorithms. Their name comes from the use of support vectors
in the decision functions.<br>
SVM is memory efficient and gives the possibility of choosing among different Kernel functions as decision
functions.<br>
The purpose of this experience is to apply SVMs on a given dataset to map its data points on the
belonging category, using both a linear and an RBF (Radial Basis Function) kernel. Different values of
regularization parameters will be introduced so that the changes on the accuracy of the model may be
observed and the best parameters identified. Moreover, the decision boundaries of the classifier will be
analyzed. At the end, K-fold Cross-Validation will be applied to improve the quality of the classification.

## HW3: Machine Learning: Deep Neural Networks
<b>Deep Learning</b> is a recent trend in machine learning that attempts to replicate the architecture of a
brain in a computer. It is believed many levels of processing exists inside the brain, in which each level
is learning features at increasing levels of abstraction.<br>
<b>Deep Neural Networks</b> (DNN) constitute a framework for different machine learning algorithms to
allow them to process complex data inputs. In DNN, each level extracts features from the output of the
previous one.<br>
Different kinds of DNN have been developed during the years and we are going to take into consideration
three of them: Traditional Neural Networks, made of hidden and fully connected layers;
<b>Convolutional Neural Networks</b> (CNN); <b>Residual Neural Networks</b> (ResNet).
