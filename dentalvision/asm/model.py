'''
Given initial knowledge of where the target object lies in an image,
the Active Shape Model algorithm amounts to a directed search of the
parameter space (Cootes 2000, 12).

See algorithm in Cootes (2000), p. 12-13.
'''
import numpy as np
from asm.fit import fit
from utils.shape import Shape
from utils.align import CoreAlign


class ActiveShapeModel(object):
    '''
    Algorithm examines a region close to the initial position. For each
    point X_i in this region, find the shape/pose parameters of the
    deformable model that fits the examined region (keeping the shape
    parameter within a 3*sqrt(eigval) bound).
    Repeat until convergence.

    in: PointDistributionModel pdmodel
        GreyLevelModel glmodel
    '''
    def __init__(self, pdmodel, glmodel):
        self.pdmodel = pdmodel
        self.glmodel = glmodel
        self.aligner = CoreAlign()

    def iterate(self, region):
        '''
        Perform the Active Shape Model algorithm
        '''
        pass

    def transform(self, Tx, Ty, s, theta, b):
        '''
        Transform the model to the image by inserting the most suitable
        pose and shape parameters
        '''
        mode = self.pdmodel.deform(b)
        return self.aligner.transform(mode, Tx, Ty, s, theta)

    def constrain(self, vector):
        '''
        All elements of the vector should agree to the following constraint:
            |v_i| < 3*sqrt(eigenval_i)
        '''
        uplimit = 3*np.sqrt(self.pdmodel.eigenvalues)
        lowlimit = -1*uplimit
        vector[vector > uplimit] = uplimit[np.where(vector > uplimit)]
        vector[vector < lowlimit] = lowlimit[np.where(vector < lowlimit)]
        return vector
