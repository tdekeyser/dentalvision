import numpy as np

from utils.structure import Shape
from utils.profile import Profile

import matplotlib.pyplot as plt


class Examiner(object):
    '''
    Class that uses the grey-level model to examine model points
    and return a set of new points that are a better fit according
    to a grey-level analysis.

    in: GreyLevelModel glmodel
    '''
    def __init__(self, glmodel):
        self.glmodel = glmodel

    def set_image(self, image):
        # transpose image for correct placement of x and y
        self.image = image.T

    def examine(self, model_points, t=0):
        '''
        Examines points normal to model points and compare its grey-levels
        with the grey-level model.

        in: matrix of pixels image
            GreyLevelModel glmodel
            array of model points (x1, x2, ..., xN, y1, y2,..., yN)
            int t amount of pixels examined either side of the normal (t > k)
        out: array of new points (x1, x2, ..., xN, y1, y2,..., yN)
        '''
        if not isinstance(model_points, Shape):
            model_points = Shape(model_points)
        amount_of_points = model_points.length
        new_points = np.zeros((2, amount_of_points))

        for m in range(amount_of_points):
            # get point and next
            points = (model_points.get(m-2), model_points.get(m-1), model_points.get(m))
            # set model index (for mean/cov)
            self.glmodel.set_evaluation_index(m)
            # get point that best matches gray levels
            new_points[:, m-1] = self.get_best_match(points, t=t)

        #### Plot for TESTING
        plt.plot(model_points.y, model_points.x, marker='o', color='r')
        plt.scatter(new_points[1,:], new_points[0,:])
        plt.show()
        ####

        return np.hstack(new_points)

    def get_best_match(self, points, t=0):
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
        best_match = points[1]
        point_profile = self.glmodel.profile(self.image, points)
        distance = self.glmodel.evaluate(point_profile)

        # determine t points along the normal for comparison
        profile = Profile(t)
        points_along_normal = profile.sample(points)

        # determine grey-level profiles of all found points
        for i in range(len(points_along_normal)):
            n_point = points_along_normal[i]
            normPoints = (points_along_normal[i], n_point, points_along_normal[i-2])

            profile = self.glmodel.profile(self.image, normPoints)

            # evaluate the profile and choose the optimal one
            new_distance = self.glmodel.evaluate(profile)
            if new_distance < distance:
                best_match = n_point
                distance = new_distance

        #### Plot for TESTING
        x = [px for px, py in points_along_normal]
        y = [py for px, py in points_along_normal]
        plt.scatter(y, x, color='y')
        ####

        return best_match
