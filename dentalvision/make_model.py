import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess.gpa import gpa


def main():
    path = '../Project Data/_Data/Landmarks/original/'
    # find all shapes in target directory
    shapes = [np.array(load(path + s)) for s in os.listdir(path) if s.endswith('.txt')]
    
    # perform gpa
    mean, aligned = gpa(shapes)
    print 'mean', mean
    
    plt.plot(mean[0,:], mean[1,:])
    for a in aligned:
        plt.plot(a[0,:], a[1,:])

    axes = plt.gca()
    axes.set_xlim([-1,1])
    plt.show()


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

