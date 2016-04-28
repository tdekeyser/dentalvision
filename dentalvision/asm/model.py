'''
Given initial knowledge of where the target object lies in an image,
the Active Shape Model algorithm amounts to a directed search of the
parameter space (Cootes 2000, 12).

See algorithm in Cootes (2000), p. 12-13.
'''
import numpy as np
from asm.fit import Fitter, Aligner
from asm.examine import Examiner
from utils.structure import Shape


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
        # initialise examining/fitting/aligning classes
        self.fitter = Fitter(pdmodel)
        self.examiner = Examiner(glmodel)
        self.aligner = Aligner()

    def iterate(self, image, region, t=10):
        '''
        Perform the Active Shape Model algorithm.

        in: array of coordinates that gives a rough estimation of the target
                in form (x1, ..., xN, y1, ..., yN)
            int t amount of pixels to be examined on each side of the normal of
                each point during an iteration
        '''
        # initialise the mean at the center of the region
        region = Shape(region)
        self.examiner.set_image(image)
        # get pose parameters to init in region
        Tx, Ty, s, theta = self.aligner.get_pose_parameters(self.pdmodel.mean, region)
        # init model mean inside region
        shape_points = self.aligner.transform(self.pdmodel.mean, Tx, Ty, s, theta)

        # examine t pixels on the normals of all points in the model (t > k)
        examinated_shape = self.examiner.examine(shape_points.vector, t=t)
        # find the best parameters to fit the model to the examined points
        Tx, Ty, s, theta, c = self.fitter.fit(examinated_shape)
        # transform the model according to the parameters
        fitted_shape = self.transform(Tx, Ty, 1, theta, c)

        return fitted_shape

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
