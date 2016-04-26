import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pdm.model import create_pdm
from glm.model import create_glm
from asm.model import ActiveShapeModel
from asm.fit import Aligner, fit
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


def run():
    # 1. POINT DISTRIBUTION MODEL
    pdmodel, landmarks = pointdistributionmodel()

    # 2. GRAYSCALE MODEL
    glmodel = grayscalemodel()

    # target_landmark = landmarks[0]
    # mean = pdmodel.mean
    # target = Shape(target_landmark)

    # # test fit function
    # Tx, Ty, s, theta, c = fit(pdmodel, target_landmark)

    # asm = ActiveShapeModel(pdmodel)
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
