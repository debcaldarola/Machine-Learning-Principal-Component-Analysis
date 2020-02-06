from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

N = 1087
DIM = 154587

def my_projection(X_std, X_t, PCA_components):
    return np.dot(X_std-X_t.mean_, PCA_components.T)

def my_reprojection(projected_data, X_t, PCA_components):
    return np.dot(projected_data, PCA_components) + X_t.mean_

def main():

    path = "../PACS_homework/*/*.jpg"
    images = glob.glob(path)

    y = np.zeros(N)
    X = np.zeros((N, DIM))

    #DATA PREPARATION
    i=0
    for name in images:
        img_data = np.asarray(Image.open(name))
        x = img_data.ravel()        #vectorial representation
        X[i] = x
        #assignment of ordinal labels to samples
        if "dog" in name:
            y[i] = 1
        elif "guitar" in name:
            y[i] = 2
        elif "house" in name:
            y[i] = 3
        elif "person" in name:
            y[i] = 4
        i = i+1

    #PRINCIPAL COMPONENT VISUALIZATION
    #Standardization of X
    X_std = (X- np.mean(X, axis=0))/np.std(X, axis=0)

    #PCA computation
    X_train = PCA().fit(X_std)
    pca_first60 = X_train.components_[0:60, :]
    pca_first6 = pca_first60[0:6,:]
    pca_first2 = pca_first6[0:2,:]
    pca_last6 = X_train.components_[-6:,:]

    #Data projection onto components
    projected_first2 = my_projection(X_std, X_train, pca_first2)
    projected_first6 = my_projection(X_std, X_train, pca_first6)
    projected_first60 = my_projection(X_std, X_train, pca_first60)
    projected_last6 = my_projection(X_std, X_train, pca_last6)

    #Re-projection onto original space
    reprojected_first2 = my_reprojection(projected_first2, X_train, pca_first2)
    reprojected_first6 = my_reprojection(projected_first6, X_train, pca_first6)
    reprojected_first60 = my_reprojection(projected_first60, X_train, pca_first60)
    reprojected_last6 = my_reprojection(projected_last6, X_train, pca_last6)

    reprojected_first2 = reprojected_first2 * np.std(X, axis=0) + np.mean(X, axis=0)
    reprojected_first6 = reprojected_first6 * np.std(X, axis=0) + np.mean(X, axis=0)
    reprojected_first60 = reprojected_first60 * np.std(X, axis=0) + np.mean(X, axis=0)
    reprojected_last6 = reprojected_last6 * np.std(X, axis=0) + np.mean(X, axis=0)

    #Variance

    # explained_variance = np.var(projected_first2, axis=0)
    # explained_variance_ratio = explained_variance / np.sum(explained_variance)
    # print('First 2 PC variance: ', np.cumsum(explained_variance_ratio))
    #
    # explained_variance = np.var(projected_first6, axis=0)
    # explained_variance_ratio = explained_variance / np.sum(explained_variance)
    # print('First 6 PC variance: ', np.cumsum(explained_variance_ratio))
    #
    # explained_variance = np.var(projected_first60, axis=0)
    # explained_variance_ratio = explained_variance / np.sum(explained_variance)
    # print('First 60 PC variance: ', np.cumsum(explained_variance_ratio))
    #
    # explained_variance = np.var(projected_last6, axis=0)
    # explained_variance_ratio = explained_variance / np.sum(explained_variance)
    # print('Last 6 PC variance: ', np.cumsum(explained_variance_ratio))

    explained_variance = np.var(PCA().fit_transform(X_std), axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    print('Variance: ', np.cumsum(explained_variance_ratio))
    print('Last six PC:', explained_variance_ratio[1081:1087])
    print('First six PC: ', explained_variance_ratio[0:6])

    plt.subplot(1,1,1)
    plt.plot(np.cumsum(explained_variance_ratio), label='Cumulative Explained Variance')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.grid(b=True, axis='both')
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% Explained Variance')
    plt.axhline(y=0.90, color='c', linestyle='--', label='90% Explained Variance')
    plt.axhline(y=0.50, color='r', linestyle='--', label='50% Explained Variance')
    plt.legend(loc='best')
    plt.show()
    plt.close()
    plt.subplot(1,1,1)
    plt.plot(np.cumsum(explained_variance_ratio), label='Cumulative Explained Variance')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.grid(b=True, axis='both')
    plt.xlim(0, 60)
    plt.axhline(y=0.50, color='r', linestyle='--', label='50% Explained Variance')
    plt.legend(loc='best')
    plt.show()
    plt.close()

    #Plot the reduced images

    img2 = np.reshape(reprojected_first2[0], (227,227,3)).astype(int)
    img6 = np.reshape(reprojected_first6[0], (227,227,3)).astype(int)
    img60 = np.reshape(reprojected_first60[0], (227,227,3)).astype(int)
    img_last6 = np.reshape(reprojected_last6[0], (227, 227, 3)).astype(int)
    original_img = np.reshape(X[0], (227,227,3)).astype(int)

    columns = 2
    rows = 3

    plt.subplot(rows, columns, 1)
    plt.title('Original image')
    plt.imshow(original_img)

    plt.subplot(rows, columns, 1)
    plt.title('Reconstructed image, first 2 PC')
    plt.imshow(img2)
    plt.show()

    plt.subplot(rows, columns, 1)
    plt.title('Reconstructed image, first 6 PC')
    plt.imshow(img6)
    plt.show()

    plt.subplot(rows, columns, 1)
    plt.title('Reconstructed image, first 60 PC')
    plt.imshow(img60)
    plt.show()

    plt.subplot(rows, columns, 1)
    plt.title('Reconstructed image, last 6 PC')
    plt.imshow(img_last6)

    plt.show()
    plt.close()

    #Scattered plots

    X_t = PCA(n_components=2).fit_transform(X_std)
    plt.title('1st and 2nd Principal Components')
    plt.scatter(X_t[:, 0], X_t[:, 1], c=y)
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.show()
    plt.close()

    X_t = projected_first60
    plt.title('3rd and 4th Principal Components')
    plt.scatter(X_t[:, 2], X_t[:, 3], c=y)
    plt.xlabel('3rd Principal Component')
    plt.ylabel('4th Principal Component')
    plt.show()
    plt.close()

    plt.title('10th and 11th Principal Components')
    plt.scatter(X_t[:, 9], X_t[:, 10], c=y)
    plt.xlabel('10th Principal Component')
    plt.ylabel('11th Principal Component')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

