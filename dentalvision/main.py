import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pdm.model import create_pdm
from glm.model import create_glm
from asm.model import ActiveShapeModel
from asm.fit import Fitter
from utils.structure import Shape
from utils import plot


def pointdistributionmodel():
    '''
        Create model of shape from input landmarks
    '''
    # Create deformable model based on landmark data
    landmarkpaths = [
        '../Project Data/_Data/Landmarks/original/',
        # '../Project Data/_Data/Landmarks/mirrored/'
        ]
    # find all shapes in target directory
    shapes = []
    for path in landmarkpaths:
        shapes += [load(path + s) for s in os.listdir(path) if s.endswith('.txt')]

    return create_pdm(shapes), np.asarray(shapes)


def grayscalemodel(k=0):
    '''
        Create a model of the local gray levels throughout the images.

        in: int k amount of pixels examined on each side of the normal
        out: GrayLevelModel instance
    '''
    imdir = '../Project Data/_Data/Radiographs/'
    shapedir = '../Project Data/_Data/Landmarks/original/'

    # load and convert images to grayscale
    images = [cv2.cvtColor(cv2.imread(imdir + im), cv2.COLOR_BGR2GRAY) for im in os.listdir(imdir) if im.endswith('.tif')]
    shapes = []
    for i in range(len(images)):
        shapes.append([load(shapedir + s) for s in os.listdir(shapedir) if 'landmarks'+str(i+1)+'-' in s])

    return create_glm(np.asarray(images), np.asarray(shapes), k=k)


def run():
    # 1. POINT DISTRIBUTION MODEL
    pdmodel, landmarks = pointdistributionmodel()
    print '***Point-distribution model created.***'

    # 2. GRAYSCALE MODEL
    glmodel = grayscalemodel(k=6)
    print '***Grey-level model created.***'

    # 3. ACTIVE SHAPE MODEL
    activeshape = ActiveShapeModel(pdmodel, glmodel)
    print '***Active Shape Model initiated.***'

    # get greyscale image
    image = cv2.cvtColor(cv2.imread('../Project Data/_Data/Radiographs/01.tif'), cv2.COLOR_BGR2GRAY)
    target = landmarks[2]
    new_fit = activeshape.iterate(image, target, t=15)

    # plot.render_image(image, target, title='Target')
    plot.render_image(image, new_fit, title='Result after examination and fitting.')


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
