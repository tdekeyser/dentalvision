import os
import cv2
import numpy as np

from pdm.model import create_pdm
from glm.model import create_glm
from asm.model import ActiveShapeModel
from utils import plot
from utils.multiresolution import gaussian_pyramid


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


def grayscalemodel(images, k=0, reduction_factor=1):
    '''
    Create a model of the local gray levels throughout the images.

    in: list of np array; images
        int k; amount of pixels examined on each side of the normal
        int reduction_factor; the change factor of the shape coordinates
    out: GrayLevelModel instance
    '''
    shapedir = '../Project Data/_Data/Landmarks/original/'

    shapes = []
    for i in range(len(images)):
        shapes.append([load(shapedir + s) for s in os.listdir(shapedir) if 'landmarks'+str(i+1)+'-' in s])

    return create_glm(images, np.asarray(shapes)/reduction_factor, k=k)


def grayscalemodel_pyramid(levels=0, k=0):
    '''
    Create grayscale models for different levels of subsampled images.
    Each subsampling is done after  by removing half the pixels.

    in: int levels amount of levels in the pyramid
        int k amount of pixels examined on each side of the normal
    out: list of graylevel models
    '''
    imdir = '../Project Data/_Data/Radiographs/'
    # load and convert images to grayscale
    images = np.asarray([cv2.cvtColor(cv2.imread(imdir + im), cv2.COLOR_BGR2GRAY) for im in os.listdir(imdir) if im.endswith('.tif')])
    # create Gaussian pyramids for each image
    multi_res = np.asarray([gaussian_pyramid(images[i], levels=levels) for i in range(images.shape[0])])

    # create list of gray-level models
    glmodels = []
    for l in range(levels):
        glmodels.append(grayscalemodel(multi_res[:, l], k=k, reduction_factor=2**l))
        print '---Created gray-level model of level ' + str(l)

    return glmodels


def load(path):
    '''
    Load and parse the data from path, and return arrays of x and y coordinates.
    Data should be in the form (x1, y1)

    in: String pathdirectory
    '''
    data = np.loadtxt(path)
    x = data[::2, ]
    y = data[1::2, ]
    return np.hstack((x, y))


def run():
    # 1. POINT DISTRIBUTION MODEL
    print '***Building Point-Distribution Model...'
    pdmodel, landmarks = pointdistributionmodel()
    print 'Done.'

    # 2. GRAYSCALE MODELs using multi-resolution images
    print '***Building Gray-Level Model pyramid...'
    glmodel_pyramid = grayscalemodel_pyramid(k=8, levels=4)
    print 'Done.'

    # 3. ACTIVE SHAPE MODEL
    print '***Initialising Active Shape Model...'
    activeshape = ActiveShapeModel(pdmodel, glmodel_pyramid)
    print 'Done.'

    # get greyscale image
    image = cv2.cvtColor(cv2.imread('../Project Data/_Data/Radiographs/01.tif'), cv2.COLOR_BGR2GRAY)
    # image = cv2.cvtColor(cv2.imread('../Project Data/_Data/Segmentations/01-5.png'), cv2.COLOR_BGR2GRAY)

    target = landmarks[2]
    # search and fit image
    new_fit = activeshape.multi_resolution_search(image, target, t=16, max_level=3, max_iter=5)
    # plot result
    plot.render_image(image, new_fit, title='result new fit from main.py')

    # plot.render_image(image, target, title='Target')
    # plot.render_image(image, new_fit, title='Result after examination and fitting.')


if __name__ == '__main__':
    run()
