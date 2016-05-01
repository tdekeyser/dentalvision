'''
Given initial knowledge of where the target object lies in an image,
the Active Shape Model algorithm amounts to a directed search of the
parameter space (Cootes 2000, 12).

See algorithm in Cootes (2000), p. 12-13.
'''
import numpy as np

from asm.fit import Fitter
from asm.examine import Examiner
from utils.align import Aligner
from utils.structure import Shape
from utils import plot


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

    def iterate(self, image, region, t=0):
        '''
        Perform the Active Shape Model algorithm.

        in: array of coordinates that gives a rough estimation of the target
                in form (x1, ..., xN, y1, ..., yN)
            int t amount of pixels to be examined on each side of the normal of
                each point during an iteration (t>k)
        '''
        if not isinstance(region, Shape):
            region = Shape(region)

        # get initial parameters
        pose_para = self.aligner.get_pose_parameters(self.pdmodel.mean, region)
        b = np.zeros(self.pdmodel.dimension)

        # align model mean with region
        shape_points = self.aligner.transform(self.pdmodel.mean, pose_para)
        #plot.render_image(image, shape_points, title='Model mean initialisation')

        # set initial fitting pose parameters and examiner image
        self.fitter.start_pose = pose_para
        self.examiner.set_image(image)

        i = 0
        shape_difference = 1
        while np.sum(shape_difference) > 0:
            # examine t pixels on the normals of all points in the model
            adjustments = self.examiner.examine(shape_points, t=t)
            # find the best parameters to fit the model to the examined points
            pose_para, c = self.fitter.fit(shape_points, adjustments)

            # add constraints to the shape parameter
            c = self.constrain(c)

            # transform the model according to the new parameters
            shape_points = self.transform(pose_para, c)

            # look for change in shape parameter and stop if necessary
            shape_difference = c - b
            b = c

            # keep iteration count
            i += 1
            print '**** ITER ---', str(i)
            print '(constr shape param)', c[:8]
            print '(eigenvalues)', self.pdmodel.eigenvalues[:8]
            # #### avoid infinite loops
            if i == 20:
                break
            #####

        return shape_points

    def transform(self, pose_para, b):
        '''
        Transform the model to the image by inserting the most suitable
        pose and shape parameters
        '''
        mode = self.pdmodel.deform(b)
        # return mode
        # plot.render_shape(mode)
        return self.aligner.transform(mode, pose_para)

    def constrain(self, vector):
        '''
        Add constraints to shape parameter proportional to the
        eigenvalues of the point distribution model.
        '''
        return vector.dot(np.diag(np.sqrt(self.pdmodel.eigenvalues)))
