'''
Create plots of particular stages of the project.
'''
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
    x = model.deform(np.zeros(52))
    y = model.deform(np.zeros(52)+0.1)

    mx, my = np.split(x, 2)
    nx, ny = np.split(y, 2)

    plt.plot(mx, my)
    plt.plot(nx, ny)
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    plt.show()
