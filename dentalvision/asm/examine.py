import numpy as np

from glm.profile import Profiler
from utils.structure import Shape


class Examiner(object):
    '''
    Class that uses the grey-level model to examine model points
    and return a set of new points that are a better fit according
    to a grey-level analysis.

    in: GreyLevelModel glmodel
    '''
    def __init__(self, glmodel_pyramid):
        self.glmodel_pyramid = glmodel_pyramid
        self.profiler = Profiler()
        self.bigImage = None

    def set_image(self, image):
        # transpose image for correct placement of x and y
        self.image = image.T

    def examine(self, model_points, t=0, pyramid_level=0):
        '''
        Examines points normal to model points and compare its grey-levels
        with the grey-level model.

        in: matrix of pixels image
            array of model points (x1, x2, ..., xN, y1, ..., yN)
            int t amount of pixels examined either side of the normal (t > k)
        out: Shape with adjustments (dx, dy) to better approximate target
        '''
        if not isinstance(model_points, Shape):
            model_points = Shape(model_points)
        new_points = model_points.matrix
        # keep track of large movements
        movement = np.zeros((model_points.length, 1))
        # get greylevel model for requested pyramid level
        glmodel = self.glmodel_pyramid[pyramid_level]
        # determine reduction based on pyramid level
        reduction = 2**pyramid_level

        i = 0
        for m in range(model_points.length):
            i += 1
            # set model index (for mean/cov)
            glmodel.set_evaluation_index(m-1)
            # choose model points according to pyramid level
            prev, curr, nex = m-2, m-1, m
            reduced_points = model_points/reduction
            # get current, previous and next
            points = np.array([reduced_points.get(prev), reduced_points.get(curr), reduced_points.get(nex)])
            # get point that best matches gray levels
            new_points[:, curr], movement[curr] = self.get_best_match(glmodel, points, t=t)
        print 'Number of points examined:', str(i)

        return Shape(new_points)*reduction, movement

    def get_best_match(self, glmodel, points, t=0):
        '''
        Returns point along the normal that matches best with the grey-level
        model and is therefore a point that is supposedly closer to the
        target shape.

        in: int point index
            triple previous point, point and next point
            int t amount of pixels to be examined each side of the normal (t>k)
        out: tuple of coordinates that is the best grey-level match
        '''
        # initiate point as best match and its distance
        best_match = points[0, :]
        point_profile = glmodel.profile(self.image, points)
        distance = glmodel.evaluate(point_profile)
        # keep track of amount of movement
        nonmoving = 1
        downlim = t/4
        uplim = 3*t/4

        # determine t points along the normal for comparison
        self.profiler.reset(t)
        points_along_normal = self.profiler.sample(points)

        # determine grey-level profiles of all found points
        for i in range(len(points_along_normal)):
            n_point = points_along_normal[i-1]
            normPoints = (points_along_normal[i-2], n_point, points_along_normal[i])

            profile = glmodel.profile(self.image, normPoints)

            # evaluate the profile and choose the optimal one
            new_distance = glmodel.evaluate(profile)
            if new_distance < distance:
                best_match = n_point
                distance = new_distance

                if (i < downlim) or (i > uplim):
                    nonmoving = 0
                else:
                    nonmoving = 1

        return best_match, nonmoving
