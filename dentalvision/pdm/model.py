'''
Create a deformable model by preprocessing landmark data
using GPA and PCA. Based on shape parameters b and eigenvectors P,
the model can construct variations of a mean using the equation:

    x = mean + P*b (see Cootes p.6 eq.2)
    OR
    x = mean + P*diag(eigenvals)*b (see Blanz p.2 eq.7) (=scaled eigenvectors)

Used in the Active Shape Model environment. The Active Shape Model
is then the fit function that iteratively searches for the best
parameters.
'''
import os
import numpy as np
from gpa import gpa
from pca import pca
from plots.plot import plot


def create_pdm(paths):
    '''
    Create a new point distribution model based on landmark data.

    Step 1: Generalised Procrustes Analysis on the landmark data
    Step 2: Principal Component Analysis on the GPAed landmark data
            This process will return an amount of eigenvectors and
            a deformable model that can construct deviations from
            the mean image.
    Step 3: Create a deformable model from the processed data

    In: list of directories of the landmark data
    Out: DeformableModel instance created with preprocessed data.
    '''
    # find all shapes in target directory
    shapes = []
    for path in paths:
        shapes += [load(path + s) for s in os.listdir(path) if s.endswith('.txt')]

    # perform gpa
    mean, aligned = gpa(np.asarray(shapes))
    plot('gpa', mean, aligned)

    # perform PCA
    eigenvalues, eigenvectors, mean = pca(aligned, mean=mean, max_variance=0.98)
    plot('eigenvectors', mean, eigenvectors)

    # create PointDistributionModel instance
    model = PointDistributionModel(eigenvalues, eigenvectors, mean)
    plot('deformablemodel', model)

    return model, shapes


def load(path):
    '''
    load and parse the data, and return arrays of x and y coordinates
    '''
    data = np.loadtxt(path)
    x = data[::2, ]
    y = data[1::2, ]
    return np.hstack((x, y))


class PointDistributionModel(object):
    '''
    Model created based on a mean image and a matrix
    of eigenvectors and corresponding eigenvalues.
    Based on shape parameters, it is able to create a
    variation on the mean shape.
    '''
    def __init__(self, eigenvalues, eigenvectors, mean):
        self.dimension = eigenvalues.size
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.mean = mean
        self.length = mean.size

    def deform(self, shape_param):
        '''
        Reconstruct and image based on principal components and a set of
        parameters that define a deformable model (see Cootes p. 6 eq. 2)

        in: Tx1 vector deformable model b
        out: 1xC deformed image
        '''
        return self.mean + self.eigenvectors.dot(shape_param).T
