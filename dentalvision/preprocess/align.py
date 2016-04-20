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


class ShapeAligner(object):
    '''
    Aligns shapes according to a target shape. Contains method
    to set a mean shape. Main method is the align method that takes a shape
    as input and aligns it to the target shape according to the
    following steps:

    Step 1 - translate the shape such that its centroid is situated in
        the origin (0,0)
    Step 2 - scale and normalize the image
    Step 3 - rotate the image in order to align it with the target
    '''
    def __init__(self, target_shape):
        self.set_target_shape(target_shape)

    def set_target_shape(self, shape):
        self.target_shape = Shape(shape).matrix

    def align(self, shape):
        '''
        Align shape with the target shape and return the aligned shape.
        All arrays in form (x1, ..., xC, ..., y1, ..., yC)
        In: 1xC array shape
        Out: 1xC aligned shape
        '''
        stacked_shape = Shape(shape).matrix
        # perform aligning
        translated = self.__translate(stacked_shape)
        scaled = self.__scale(translated)
        aligned = self.__rotate(scaled, self.target_shape)
        return np.hstack((aligned[0], aligned[1]))

    def __compute_centroid(self, x, y):
        '''Compute the centroid: the average of an array of coordinates'''
        return (np.sum(x)/x.shape, np.sum(y)/y.shape)

    def __translate(self, shape):
        '''
        Move all shapes to a common center, most likely the origin (0,0)

        In: array x
            array y
        Out = array, array
        '''
        x, y = shape[0, :], shape[1, :]
        # compute centroid
        centr_x, centr_y = self.__compute_centroid(x, y)
        # translate w.r.t. centroid
        return np.array([x - centr_x, y - centr_y])

    def __scale(self, shape):
        '''
        Perform isomorphic scaling on input arrays

        out: scaled array x, array y
        '''
        x, y = shape[0, :], shape[1, :]
        return np.array([x/np.linalg.norm(x), y/np.linalg.norm(y)])

    def __rotate(self, shape, target_shape):
        '''
        Rotate shape such that it aligns with the target shape

        out: rotated Rx2 shape
        '''
        # perform singular value decomposition (svd) to get U, S, V'
        u, s, v = np.linalg.svd(target_shape.dot(np.transpose(shape)))
        # multiply VU' with shape to get the rotation
        vu = np.transpose(v).dot(np.transpose(u))
        return vu.dot(shape)


class AlignmentFinder(ShapeAligner):
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
