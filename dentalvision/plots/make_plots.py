'''
Create plots of particular stages of the project.
'''
import numpy as np
import matplotlib.pyplot as plt


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

    for i in range(6):
        vec = eigenvectors[:,i].T
        axes.arrow(0, 0, vec[0], vec[1], fc='k', ec='k')

    plt.show()
