from PIL import Image
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


N = 1087
DIM = 154587

def main():
    path = "../PACS_homework/*/*.jpg"
    images = glob.glob(path)

   # y = np.zeros(N) #Labels: 1=dog, 2=guitar, 3=house, 4=person
    y = np.zeros(N)
    X = np.zeros((N, DIM))

    # DATA PREPARATION
    i = 0
    for name in images:
        img_data = np.asarray(Image.open(name))
        x = img_data.ravel()  # vectorial representation
        X[i] = x
        # assignment of ordinal labels to samples
        if "dog" in name:
            y[i] = 1
        elif "guitar" in name:
            y[i] = 2
        elif "house" in name:
            y[i] = 3
        elif "person" in name:
            y[i] = 4
        i = i + 1

    # CLASSIFICATION
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Samples split in test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #test_size=0.1

    # Naive Bayes classifier - Training and testing with Gaussian distribution
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    acc = accuracy_score(y_test, y_pred, normalize=True)
    print('Predicted labels on the test set:\n', y_pred, sep=" ")
    print('True labels on the test set:\n', y_test, sep=" ")
    print('\nAccuracy on unmodified images: ', acc*100) #74.31192660550458%

    # PCA algorithm
    X_trans = PCA(4).fit(X_std)

    # First 2 principal components

    pca2 = X_trans.components_[0:2, :]
    X_transf2 = np.dot(X_std - X_trans.mean_, pca2.T)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_transf2, y, test_size=0.1)
    y2_pred = gnb.fit(X2_train, y2_train).predict(X2_test)
    acc2 = accuracy_score(y2_test, y2_pred, normalize=True)
    print('Predicted labels on the test set:\n', y2_pred, sep=" ")
    print('True labels on the test set:\n', y2_test, sep=" ")
    print('\nAccuracy on images projected onto the first two components: ', acc2 * 100) #66.05504587155964%

    # 3rd and 4th principal components

    pca4 = X_trans.components_[2:4,:]
    X_transf4 = np.dot(X_std-X_trans.mean_, pca4.T)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X_transf4, y, test_size=0.1)
    y4_pred = gnb.fit(X4_train, y4_train).predict(X4_test)
    acc4 = accuracy_score(y4_test, y4_pred, normalize=True)
    print('Predicted labels on the test set:\n', y4_pred, sep=" ")
    print('True labels on the test set:\n', y4_test, sep=" ")
    print('\nAccuracy on images projected onto the 3rd and 4th components: ', acc4 * 100) #44.95412844036697%


    # PLOT DECISION BOUNDARIES
    # Computation of decision surfaces
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, yy1 = np.meshgrid(np.arange(x_min, x_max, 127),
                           np.arange(y_min, y_max, 245))

    x_min, x_max = X_transf2[:, 0].min() - 1, X_transf2[:, 0].max() + 1
    y_min, y_max = X_transf2[:, 1].min() - 1, X_transf2[:, 1].max() + 1
    xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, 127),
                           np.arange(y_min, y_max, 245))

    x_min, x_max = X_transf4[:, 0].min() - 1, X_transf4[:, 0].max() + 1
    y_min, y_max = X_transf4[:, 1].min() - 1, X_transf4[:, 1].max() + 1
    xx3, yy3 = np.meshgrid(np.arange(x_min, x_max, 127),
                           np.arange(y_min, y_max, 245))

    # Plot of decision regions
    i = 0
    for title, xx, yy, X, Y, label in zip(['Original data', '1st and 2nd PC', '3rd and 4th PC'], [xx1, xx2, xx3], [yy1, yy2, yy3],
                                            [X[:,0], X_transf2[:, 0], X_transf4[:, 0]], [X[:,1], X_transf2[:, 1], X_transf4[:, 1]],
                                            [['', ''], ['1st PC', '2nd PC'], ['3rd PC', '4th PC']]):
        Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        i = i + 1
        plt.contour(xx, yy, Z, levels=3, cmap=plt.cm.Paired)
        plt.contourf(xx, yy, Z, alpha=0.5, levels=3, cmap=plt.cm.Paired)
        plt.scatter(X, Y, c=y, edgecolor='k', cmap=plt.cm.Paired)
        plt.title(title)
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.show()

if __name__=='__main__':
    main()

