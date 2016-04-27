import numpy as np

from utils.shape import Shape
from utils.profile import extract_profile, Profile


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

    def examine(self, model_points, t=3):
        '''
        Examines points normal to model points and compare its grey-levels
        with the grey-level model.

        in: matrix of pixels image
            GreyLevelModel glmodel
            array of model points (x1, x2, ..., xN, y1, y2,..., yN)
            int t amount of pixels examined either side of the normal (t > k)
        out: array of new points (x1, x2, ..., xN, y1, y2,..., yN)
        '''
        model_points = Shape(model_points)
        amount_of_points = model_points.length
        new_points = np.zeros((2, amount_of_points))

        for m in range(amount_of_points):
            # get point and next
            point = model_points.get(m)
            next_point = model_points.get(m-1)
            self.glmodel.set_evaluation_index(m)
            new_points[:, m] = self.get_best_match(m, point, next_point, t=t)

        return np.hstack(new_points)

    def get_best_match(self, point, next_point, t=3):
        '''
        Returns point along the normal that matches best with the grey-level
        model and is therefore a point that is supposedly closer to the
        target shape.

        in: int point index
            tuple point and next point
            int t amount of pixels to be examined each side of the normal (t>k)
        out: tuple of coordinates that is the best grey-level match
        '''
        # slice frame around model point
        x, y = point
        frame = self.image[x-t:x+t, y-t:y+t]

        # determine t points along the normal
        profile = Profile(point, next_point)
        profile_coordinates = profile.get_closest(frame, t)
        profile_greys = profile.get_profile(frame, t)

        # initiate best match
        best_match = point
        current_dist = self.glmodel.evaluate(profile_greys)

        # determine grey-level profiles of all found points
        for i in range(len(profile_coordinates)):
            coord = profile_coordinates[i]
            next_coord = profile_coordinates[i-1]
            profile = extract_profile(self.image, (coord, next_coord), k=t)
            # evaluate the profile
            M = self.glmodel.evaluate(profile)
            if M < current_dist:
                best_match = coord
                current_dist = M

        return best_match
