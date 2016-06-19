import numpy as np

from glm.profile import Profile
from utils.structure import Shape
from utils import plot


class Examiner(object):
    '''
    Class that uses the grey-level model to examine model points
    and return a set of new points that are a better fit according
    to a grey-level analysis.

    in: GreyLevelModel glmodel
    '''
    def __init__(self, glmodel_pyramid):
        self.glmodel_pyramid = glmodel_pyramid
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
        # get greylevel model for requested pyramid level
        glmodel = self.glmodel_pyramid[pyramid_level]
        # determine reduction based on pyramid level
        reduction = 2**pyramid_level
        # adjust number of examined on each side of the normal acc. to level
        t = t/(pyramid_level+1)

        i = 0
        for m in range(model_points.length):
            i += 1
            # set model index (for mean/cov)
            glmodel.set_evaluation_index(m)
            # choose model points according to pyramid level
            prev, curr, nex = m-2, m-1, m
            reduced_points = model_points/reduction
            # get current, previous and next
            points = np.array([reduced_points.get(prev), reduced_points.get(curr), reduced_points.get(nex)])
            # get point that best matches gray levels
            new_points[:, curr] = self.get_best_match(glmodel, points, t=t)*reduction
        print 'Number of points examined:', str(i)

        #### Plot for TESTING
        # plot.render_shape_to_image(self.image, reduced_points, color=(255, 0, 0))
        # plot.render_shape_to_image(self.bigImage, np.hstack(new_points), color=(255, 0, 0))
        ####

        return Shape(new_points)

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

        # determine t points along the normal for comparison
        profile = Profile(t)
        points_along_normal = profile.sample(points)

        # determine grey-level profiles of all found points
        for i in range(len(points_along_normal)-1):
            n_point = points_along_normal[i-1]
            normPoints = (points_along_normal[i-2], n_point, points_along_normal[i])

            profile = glmodel.profile(self.image, normPoints)

            # evaluate the profile and choose the optimal one
            new_distance = glmodel.evaluate(profile)
            if new_distance < distance:
                best_match = n_point
                distance = new_distance

        return best_match
