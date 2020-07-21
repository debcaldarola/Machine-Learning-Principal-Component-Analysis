from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import  matplotlib.pyplot as plt
import numpy as np

def make_meshgrid(X0, X1, h):
    """
    Create a mesh of points to be plotted

    :param X0: data to base x-axis meshgrid on
    :param X1:  data to base y-axis meshgrid on
    :param h: stepsize for meshgrid
    :return: nd arrays
    """
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_decision_boundaries(ax, clf, xx, yy, **params):
    """
    Plot decision boundaries of the given classifier
    :param ax: axes of the matplotlib object
    :param clf: classifier
    :param xx: nd array
    :param yy: nd array
    :param params: parameters to pass to countours
    :return: contours
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    contours = ax.contourf(xx, yy, Z, **params)
    return contours

def main():

    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # first 2 features
    y = iris.target

    # Split samples in train, validation and test sets (proportion 5:2:3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

    # Linear and RBF kernels (3.1)
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000] # SVM regularization parameter
    h=.02

    kernels = ('linear', 'rbf')

    for k in kernels:

        figure, axes = plt.subplots(2, 4)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        scores = np.zeros(len(C))
        i = 0
        max_score = -1

        for c, ax in zip(C, axes.flatten()):
            # Train SVM
            clf = svm.SVC(kernel=k, C=c, gamma='auto')
            clf.fit(X_train, y_train)
            scores[i] = clf.fit(X_train, y_train).score(X_val, y_val)
            if scores[i] > max_score:
                max_score = scores[i]
                best_c = c
            i += 1

            # Plot decision boundaries
            X0, X1 = X_train[:,0], X_train[:,1]
            xx, yy = make_meshgrid(X0, X1, h)
            plot_decision_boundaries(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel('Sepal length')
            ax.set_ylabel('Sepal width')
            title = 'C=' + str(c)
            ax.set_title(title)
        plt.show()

        # Evaluation on Validation set
        print('Scores on validation:')
        print(scores)
        #[0.40625 0.59375 0.8125  0.84375 0.78125 0.78125 0.78125]
        #[0.40625 0.40625 0.75    0.84375 0.78125 0.8125  0.84375]

        plt.plot(C, scores)
        plt.semilogx()
        plt.grid(b=True, axis='both')
        plt.axhline(y=scores.max(), color='r', linestyle='--', label='Maximum score on validation set')
        plt.legend(loc="best")
        title = 'Accuracy on validation, ' + k + ' kernel'
        plt.title(title)
        plt.show()

        # Evaluation on test set
        clf = svm.SVC(kernel=k, C=best_c, gamma='auto')
        acc_val = clf.fit(X_train, y_train).score(X_test, y_test)
        print('Accuracy on the test set (' + k + ' kernel): %0.2f' % acc_val)
        #Accuracy on the test set (linear kernel): 0.73
        #Accuracy on the test set (rbf kernel): 0.76

        # Plot decision boundaries on the test set
        X0, X1 = X_test[:, 0], X_test[:, 1]
        xx, yy = make_meshgrid(X0, X1, h)
        fig, ax = plt.subplots(1,1)
        plot_decision_boundaries(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_title('Decision boundaries on the test set (' + k + ' kernel)')
        plt.show()


    # GRID SEARCH of the best C and Gamma for RBF kernel (3.2)

    gamma = [10**-9, 10**-7, 10**-5, 10**-3, 10**-1, 10**0, 10**1]

    param_grid = [{'C' : C, 'gamma' : gamma, 'kernel' : ['rbf']}]
    clf = GridSearchCV(svm.SVC(), param_grid=param_grid, iid=True)
    clf.fit(X_train, y_train)
    print("The best parameters are %s with a score on the validation set of %0.2f.\n" % (clf.best_params_, clf.fit(X_train, y_train).score(X_val, y_val)))
    #The best parameters are {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'} with a score on the validation set of 0.78.
    # Show table
    matr = np.zeros((len(C), len(gamma)))
    i = 0

    for c in C:
        j = 0
        for g in gamma:
            classif = svm.SVC(kernel='rbf', C=c, gamma=g)
            matr[i][j] = classif.fit(X_train, y_train).score(X_val, y_val)
            j += 1
        i += 1

    print('Scores on validation set (all parameters):')
    print(matr)
    #Scores on validation set (all parameters):
    # [[0.40625 0.40625 0.40625 0.40625 0.40625 0.40625 0.40625]
    #  [0.40625 0.40625 0.40625 0.40625 0.40625 0.40625 0.40625]
    #  [0.40625 0.40625 0.40625 0.40625 0.78125 0.8125  0.65625]
    #  [0.40625 0.40625 0.40625 0.40625 0.8125  0.78125 0.8125 ]
    #  [0.40625 0.40625 0.40625 0.71875 0.8125  0.8125  0.84375]
    #  [0.40625 0.40625 0.40625 0.8125  0.78125 0.84375 0.8125 ]
    #  [0.40625 0.40625 0.71875 0.8125  0.8125  0.84375 0.75   ]]
    plt.matshow(matr)
    plt.gca().xaxis.tick_bottom()
    plt.xlabel('C')
    plt.ylabel('Gamma')
    cbar = plt.colorbar()
    cbar.set_label('Score on validation set')
    plt.yticks(np.arange(len(gamma)), gamma)
    plt.xticks(np.arange(len(C)), C)
    plt.title('Accuracy on validation set - RBF kernel')
    plt.show()

    # Evaluation of the best parameters on the test set

    best_c = clf.best_params_.get("C")
    best_gamma = clf.best_params_.get("gamma")
    #print("C: %0.2f, gamma=%0.2f" %(best_c, best_gamma))
    clf2 = svm.SVC(C=best_c, gamma=best_gamma, kernel='rbf')
    acc_val = clf2.fit(X_train, y_train).score(X_test, y_test)
    print('Accuracy on the test set (RBF kernel, best parameters): %0.2f' % acc_val) #Accuracy on the test set (RBF kernel, best parameters): 0.76

    # Plot decision boundaries on the test set
    X0, X1 = X_test[:, 0], X_test[:, 1]
    xx, yy = make_meshgrid(X0, X1, h)
    fig, ax = plt.subplots(1, 1)
    plot_decision_boundaries(ax, clf2, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('Decision boundaries on the test set (RBF kernel, best parameters)')
    plt.show()

    # K-FOLD VALIDATION (3.3)

    # Merge train and validation sets
    np.concatenate((X_train, X_val), axis=None)
    np.concatenate((y_train, y_val), axis=None)

    # Split training set into folds
    X_folds = np.array_split(X_train, 5)
    y_folds = np.array_split(y_train, 5)

    scores = np.empty((len(gamma), len(C), len(X_folds)))  #??
    avg_scores = np.empty((len(gamma), len(C)))
    max_score = -1

    for i, g in enumerate(gamma):
        for j, c in enumerate(C):
            for k in range(5):
                clf = svm.SVC(kernel='rbf', C=c, gamma=g)
                X_train = list(X_folds)
                X_test = X_train.pop(k)
                X_train = np.concatenate(X_train)
                y_train = list(y_folds)
                y_test = y_train.pop(k)
                y_train = np.concatenate(y_train)
                clf.fit(X_train, y_train)
                scores[i,j,k] = clf.fit(X_train, y_train).score(X_test, y_test)

            avg_scores[i,j] = scores[i,j].mean()    # final accuracy = avg(Round1, Round2, ... )
            if avg_scores[i,j] > max_score:
                max_score = avg_scores[i,j]
                best_c = c
                best_gamma = g

    print('Average scores:')
    print(avg_scores)
    #Average scores:
    # [[0.30095238 0.30095238 0.30095238 0.30095238 0.30095238 0.30095238
    #   0.30095238]
    #  [0.30095238 0.30095238 0.30095238 0.30095238 0.30095238 0.30095238
    #   0.30095238]
    #  [0.30095238 0.30095238 0.30095238 0.30095238 0.30095238 0.30095238
    #   0.56285714]
    #  [0.30095238 0.30095238 0.30095238 0.30095238 0.54857143 0.79428571
    #   0.83619048]
    #  [0.30095238 0.30095238 0.40952381 0.79428571 0.82190476 0.80952381
    #   0.76666667]
    #  [0.30095238 0.30095238 0.71238095 0.82190476 0.80857143 0.80857143
    #   0.78285714]
    #  [0.28761905 0.28761905 0.34190476 0.7952381  0.75333333 0.65904762
    #   0.65904762]]
    plt.matshow(avg_scores)
    plt.gca().xaxis.tick_bottom()
    plt.xlabel('C')
    plt.ylabel('Gamma')
    cbar = plt.colorbar()
    cbar.set_label('Accuracy on test set')
    plt.yticks(np.arange(len(gamma)), gamma)
    plt.xticks(np.arange(len(C)), C)
    plt.title('5-fold Validation: Accuracy on test set')
    plt.show()

    print("Best parameters: C=%0.3f, gamma=%0.3f, accuracy on test set = %0.2f" % (best_c, best_gamma, avg_scores.max())) #Best parameters: C=1000.00, gamma=0.001, accuracy on test set = 0.84

if __name__ == '__main__':
    main()