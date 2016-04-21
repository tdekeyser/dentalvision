'''
Create plots of particular stages of the project.
'''
import math
import numpy as np
import matplotlib.pyplot as plt


PLOT_STAGES = True


def plot(choice, *args):
    '''
    Plot the different stages of the project. Set PLOT_STAGES to False
    to avoid plotting.
    '''
    if not PLOT_STAGES:
        return
    else:
        if choice == 'gpa':
            return plot_gpa(*args)
        elif choice == 'eigenvectors':
            return plot_eigenvectors(*args)
        elif choice == 'deformablemodel':
            return plot_deformablemodel(*args)


def plot_gpa(mean, aligned_shapes):
    '''
    Plot the result of GPA; plot the mean and the first 10 deviating shapes
    '''
    # plot mean
    mx, my = np.split(mean, 2)
    plt.plot(mx, my, marker='o')
    # plot first i aligned deviations
    for i in range(3):
        a = aligned_shapes[i, :]
        ax, ay = np.split(a, 2)
        plt.plot(ax, ay, marker='o')
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    plt.show()


def plot_eigenvectors(mean, eigenvectors):
    '''
    Plot the eigenvectors within a mean image.
    Centroid of mean must be the origin!
    '''
    mx, my = np.split(mean, 2)
    plt.plot(mx, my, marker='o')

    axes = plt.gca()
    axes.set_xlim([-1, 1])

    for i in range(6):
        vec = eigenvectors[:, i].T
        axes.arrow(0, 0, vec[0], vec[1], fc='k', ec='k')

    plt.show()


def plot_deformablemodel(model):
    z = np.zeros(52)

    # recreate the mean
    x = model.deform(z)
    mx, my = np.split(x, 2)
    plt.plot(mx, my)

    # create variations
    for i in range(2):
        limit = 3*math.sqrt(model.eigenvalues[i])
        z[i] = -1*limit
        y = model.deform(z)
        nx, ny = np.split(y, 2)
        plt.plot(nx, ny)

    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-0.5, 0.5])
    plt.show()
