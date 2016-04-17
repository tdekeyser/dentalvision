import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess.gpa import gpa
from preprocess.pca import pca


def main():
    # do GPA
    mean, aligned = perform_gpa()
    # do PCA
    perform_pca(mean, aligned)


def perform_pca(mean, aligned_shapes):
    # perform pca
    t = np.transpose(aligned_shapes[7])
    eigenvalues, eigenvectors, m = pca(t, np.transpose(mean))
    print 'eigenvalues', eigenvalues
    print "eigenvectors", eigenvectors

    plt.plot(mean[0,:], mean[1,:])
    plt.plot(aligned_shapes[7][0,:], aligned_shapes[7][1,:])

    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1,1])

    for i in range(2):
        vec = eigenvectors[:,i].T
        print vec
        axes.arrow(0, 0, vec[0]/2, vec[1]/2, fc='k', ec='k')
        axes.arrow(0, 0, -1*vec[0]/2, -1*vec[1]/2)

    plt.show()
    return


def perform_gpa():
    path = '../Project Data/_Data/Landmarks/original/'
    # find all shapes in target directory
    shapes = [np.array(load(path + s)) for s in os.listdir(path) if s.endswith('.txt')]
    
    # perform gpa
    mean, aligned = gpa(shapes)

    plt.plot(mean[0,:], mean[1,:])
    i=0
    for a in aligned:
        plt.plot(a[0,:], a[1,:])
        i += 1
        if i == 10:
            break

    axes = plt.gca()
    axes.set_xlim([-1,1])

    plt.show()
    return mean, aligned



def load(path):
    '''
    load and parse the data, and return arrays of x and y coordinates
    '''
    data = np.loadtxt(path)
    x = data[::2,]
    y = data[1::2,]
    return x, y


if __name__ == '__main__':
    main()

