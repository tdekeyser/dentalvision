import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pdm.model import create_pdm
from glm.model import create_glm
from asm.model import ActiveShapeModel
from asm.fit import Aligner, Fitter
from utils.shape import Shape


def pointdistributionmodel():
    # Create deformable model based on landmark data
    landmarkpaths = [
        '../Project Data/_Data/Landmarks/original/',
        '../Project Data/_Data/Landmarks/mirrored/'
        ]
    # find all shapes in target directory
    shapes = []
    for path in landmarkpaths:
        shapes += [load(path + s) for s in os.listdir(path) if s.endswith('.txt')]

    return create_pdm(shapes), np.asarray(shapes)


def grayscalemodel():
    imdir = '../Project Data/_Data/Radiographs/'
    shapedir = '../Project Data/_Data/Landmarks/original/'

    images = [cv2.cvtColor(cv2.imread(imdir + im), cv2.COLOR_BGR2GRAY) for im in os.listdir(imdir) if im.endswith('.tif')]
    shapes = []
    for i in range(len(images)):
        shapes.append([load(shapedir + s) for s in os.listdir(shapedir) if 'landmarks'+str(i+1)+'-' in s])

    return create_glm(np.asarray(images), np.asarray(shapes), k=7)


def init_shape():
    center = (928, 727)
    return square(center)


def square(center):
    cx, cy = center
    left = np.asarray([(cx-5, cy-5+i) for i in range(10)])
    right = np.asarray([(cx+5, cy-5+i) for i in range(10)])
    up = np.asarray([(cx-5+i, cy+5) for i in range(10)])
    down = np.asarray([(cx-5+i, cy-5) for i in range(10)])

    one = np.hstack((left, up))
    two = np.hstack((right, down))

    stack = np.hstack((np.hstack(one), np.hstack(two)))
    x = stack[::2, ]
    y = stack[1::2, ]
    return np.hstack((x, y))


def run():
    # 1. POINT DISTRIBUTION MODEL
    pdmodel, landmarks = pointdistributionmodel()
    print '***Point-distribution model created.***'

    # 2. GRAYSCALE MODEL
    glmodel = grayscalemodel()
    print '***Grey-level model created.***'

    # 3. ACTIVE SHAPE MODEL
    activeshape = ActiveShapeModel(pdmodel, glmodel)
    print '***Active Shape Model initiated.***'

    # create some starting array
    init = init_shape()
    # get greyscale image
    image = cv2.cvtColor(cv2.imread('../Project Data/_Data/Radiographs/01.tif'), cv2.COLOR_BGR2GRAY)
    init = landmarks[0]
    new_fit = activeshape.iterate(image, init, t=10)
    print new_fit

    init = Shape(init)
    plt.plot(init.x, init.y, marker='o', color='r')
    plt.plot(new_fit.x, new_fit.y, marker='o', color='g')
    plt.show()

    # target_landmark = landmarks[0]
    # mean = pdmodel.mean
    # target = Shape(target_landmark)

    # # test fit function
    # Tx, Ty, s, theta, c = fit(pdmodel, target_landmark)

    # asm = ActiveShapeModel(pdmodel, glmodel)
    # targy = asm.transform(Tx, Ty, 1, theta, c)
    # aligned_mean = asm.transform(Tx, Ty, s, theta, np.zeros(pdmodel.dimension))
    # print targy.centroid()

    # aligner = Aligner()
    # variation = pdmodel.deform(c)

    # plt.plot(target.x, target.y, color='k', marker='o', label='target')
    # plt.plot(targy.x, targy.y, color='r', marker='o', label='result')

    # plt.scatter(aligned_mean.x, aligned_mean.y, color='y', marker='o', label='aligned_mean')

    # # plt.plot(mean.x, mean.y, color='g', marker='o', label='mean')
    # # plt.plot(variation.x, variation.y, color='r', marker='o', label='model variation')

    # plt.legend(loc='upper right')
    # plt.show()


def load(path):
    '''
    load and parse the data, and return arrays of x and y coordinates
    '''
    data = np.loadtxt(path)
    x = data[::2, ]
    y = data[1::2, ]
    return np.hstack((x, y))


if __name__ == '__main__':
    run()
