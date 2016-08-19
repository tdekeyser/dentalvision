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
from utils.multiresolution import gaussian_pyramid
# from utils import plot


class ActiveShapeModel(object):
    '''
    Algorithm examines a region close to the initial position. For each
    point X_i in this region, find the shape/pose parameters of the
    deformable model that fits the examined region (keeping the shape
    parameter within a 3*sqrt(eigval) bound).
    Repeat until convergence.

    in: PointDistributionModel pdmodel
        list of gray-level models per resolution level
    '''
    def __init__(self, pdmodel, glmodel_pyramid):
        self.pdmodel = pdmodel
        self.glmodel_pyramid = glmodel_pyramid

        # initialise examining/fitting/aligning classes
        self.fitter = Fitter(pdmodel)
        self.examiner = Examiner(glmodel_pyramid)
        self.aligner = Aligner()

    def multiresolution_search(self, image, region, t=0, max_level=0, max_iter=5, n=None):
        '''
        Perform Multi-resolution Search ASM algorithm.

        in: np array of training image
            np array region; array of coordinates that gives a rough
                estimation of the target in form (x1, ..., xN, y1, ..., yN)
            int t; amount of pixels to be examined on each side of the
                normal of each point during an iteration (t>k)
            int max_levels; max amount of levels to be searched
            int max_iter; amount to stop iterations at each level
            int n; fitting parameter
        out: Shape region; approximation of the target
        '''
        if not isinstance(region, Shape):
            region = Shape(region)

        # create Gaussian pyramid of input image
        image_pyramid = gaussian_pyramid(image, levels=len(self.glmodel_pyramid))

        # allow examiner to render the largest image (for testing)
        self.examiner.bigImage = image_pyramid[0]

        level = max_level
        max_level = True
        while level >= 0:
            # get image at level resolution
            image = image_pyramid[level]
            # search in the image
            region = self.search(image, region, t=t, level=level, max_level=max_level, max_iter=max_iter, n=n)
            # descend the pyramid
            level -= 1
            max_level = False

        return region

    def search(self, image, region, t=0, level=0, max_level=False, max_iter=5, n=None):
        '''
        Perform the Active Shape Model algorithm in input region.

        in: array image; input image
            array region; array of coordinates that gives a rough estimation
                of the target in form (x1, ..., xN, y1, ..., yN)
            int t; amount of pixels to be examined on each side of the normal
                of each point during an iteration (t>k)
            int level; level in the gaussian pyramid
            int max_iter; amount to stop iterations at each level
            int n; fitting parameter
        out: array points; approximation of the target
        '''
        if max_level:
            # get initial parameters
            pose_para = self.aligner.get_pose_parameters(self.pdmodel.mean, region)
            # set initial fitting pose parameters
            self.fitter.start_pose = pose_para
            # align model mean with region
            points = self.aligner.transform(self.pdmodel.mean, pose_para)
        else:
            points = region

        # set examiner image
        self.examiner.set_image(image)

        # perform algorithm
        i = 0
        movement = np.zeros_like(points.length)
        while np.sum(movement)/points.length <= 0.6:
            # examine t pixels on the normals of all points in the model
            # plot.render_shape_to_image(image, points)
            adjustments, movement = self.examiner.examine(points, t=t, pyramid_level=level)
            # find the best parameters to fit the model to the examined points
            pose_para, c = self.fitter.fit(points, adjustments, pyramid_level=level, n=n)
            # add constraints to the shape parameter
            c = self.constrain(c)
            # transform the model according to the new parameters
            points = self.transform(pose_para, c)

            print '**** LEVEL ---', str(level)
            print '**** ITER ---', str(i)
            print '(constr shape param)', c[:4]
            print '(pose params)', pose_para

            # keep iteration count
            i += 1
            if i == max_iter:
                break

        return points

    def transform(self, pose_para, b):
        '''
        Transform the model to the image by inserting the most suitable
        pose and shape parameters
        '''
        mode = self.pdmodel.deform(b)
        return self.aligner.transform(mode, pose_para)

    def constrain(self, vector):
        '''
        Add constraints to shape parameter proportional to the eigenvalues
        of the point distribution model. According to Cootes et al., all
        elements of the vector should agree to the following constraint:
          |v_i| < 3*sqrt(eigenval_i)
        '''
        uplimit = 3*np.sqrt(self.pdmodel.eigenvalues)
        lowlimit = -1*uplimit
        vector[vector > uplimit] = uplimit[np.where(vector > uplimit)]
        vector[vector < lowlimit] = lowlimit[np.where(vector < lowlimit)]
        return vector
