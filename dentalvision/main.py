import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess.gpa import gpa
from preprocess.pca import pca


def create_model():
    '''
    Create a model of the landmarks

    Step 1: Generalised Procrustes Analysis on the landmark data
    Step 2: Principal Component Analysis on the GPAed landmark data
            This process will return an amount of eigenvectors and
            a deformable model that can construct deviations from
            the mean image.
    '''
    # ----------------------- GPA ----------------------------#

    path = '../Project Data/_Data/Landmarks/original/'
    # find all shapes in target directory
    shapes = [load(path + s) for s in os.listdir(path) if s.endswith('.txt')]
    # perform gpa
    mean, aligned = gpa(np.asarray(shapes))
    # plot the result
    plot_gpa(mean, aligned)


    # ---------------------- PCA ----------------------------#

    eigenvalues, eigenvectors, mean = pca(aligned, mean=mean, max_variance=0.98)
    plot_eigenvectors(mean, eigenvectors)


def load(path):
    '''
    load and parse the data, and return arrays of x and y coordinates
    '''
    data = np.loadtxt(path)
    x = data[::2,]
    y = data[1::2,]
    return np.hstack((x, y))


def plot_gpa(mean, aligned_shapes):
    '''
    Plot the result of GPA; plot the mean and the first 10 deviating shapes
    '''
    # plot mean
    mx, my = np.split(mean, 2)
    plt.plot(mx, my, marker='o')
    # plot first i aligned deviations
    for i in range(3):
        a = aligned_shapes[i,:]
        ax, ay = np.split(a, 2)
        plt.plot(ax, ay, marker='o')
    axes = plt.gca()
    axes.set_xlim([-1,1])
    plt.show()


def plot_eigenvectors(mean, eigenvectors):
    '''
    Plot the eigenvectors within a mean image.
    Centroid of mean must be the origin!
    '''
    mx, my = np.split(mean, 2)
    plt.plot(mx, my, marker='o')

    axes = plt.gca()
    axes.set_xlim([-1,1])

    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:,i].T
        axes.arrow(0, 0, vec[0], vec[1], fc='k', ec='k')

    plt.show()


if __name__ == '__main__':
    create_model()

