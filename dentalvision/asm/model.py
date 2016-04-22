'''
Given initial knowledge of where the target object lies in an image,
the Active Shape Model algorithm amounts to a directed search of the
parameter space (Cootes 2000, 12).

See algorithm in Cootes (2000), p. 12-13.
'''
import numpy as np
from match import match
from alignment.shape import Shape
from alignment.align import CoreAlign


class ActiveShapeModel(object):
    '''
    Algorithm examines a region close to the initial position. For each
    point X_i in this region, find the shape/pose parameters of the
    deformable model that fits the examined region (keeping the shape
    parameter within a 3*sqrt(eigval) bound).
    Repeat until convergence.
    '''
    def __init__(self, pdmodel):
        self.pdmodel = pdmodel
        self.aligner = CoreAlign()

    def fit(self, region):
        '''
        Perform the Active Shape Model algorithm
        '''
        pass

    def transform(self, Tx, Ty, s, theta, b):
        '''
        Transform the model to the image by inserting the most suitable
        pose and shape parameters
        '''
        mode = Shape(self.pdmodel.deform(b))
        return self.aligner.transform(mode, Tx, Ty, s, theta)

    def __check_constraints(self, vector):
        '''
        All elements of the vector should agree to the following constraint:
            |v_i| < 3*sqrt(eigenval_i)
        '''
        limit = 3*np.sqrt(self.eigenvalues)
        vector[vector > limit] = limit
        vector[vector < -1*limit] = -1*limit
        return vector
