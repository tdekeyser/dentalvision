import numpy as np


class Shape(object):
    '''
    Class that represents a basic shape.
    In: arrays in form (x1, ..., xC, ..., y1, ..., yC)
    '''
    def __init__(self, array):
        self.x, self.y = np.split(array, 2)
        self.array = array
        self.matrix = np.vstack((self.x, self.y))

    def compute_centroid(self):
        '''Compute the centroid: the average of an array of coordinates'''
        return (np.sum(self.x)/self.x.shape, np.sum(self.y)/self.y.shape)


class AlignmentFinder(object):
    '''
    Class that is able to find the best parameters to initiate an
    alignment between shapes.
    Inherits form ShapeAligner.
    '''

    def get_pose_parameters(self, shape1, shape2):
        '''
        Find the pose parameters to align shape1 with shape2.
        In: 1xC array shape1, shape2
        Out: X_t, Y_t params that define the translation,
            s, defines the scaling
            delta, defines the rotation
        '''
        self.shape1 = np.vstack(np.split(shape1, 2))
        self.shape2 = np.vstack(np.split(shape2, 2))

        # TODO
        # see p. 24-25 in Cootes
        return

    def __find_scale(self):
        # TODO
        return

    def __find_rotation_angle(self):
        # TODO
        return

    def __find_translation(self):
        centroid1 = self.shape1.compute_centroid()
        centroid2 = self.shape2.compute_centroid()

        # TODO
        return
