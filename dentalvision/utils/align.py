import math
import numpy as np

from utils.structure import Shape


class CoreAlign(object):
    '''
    Methods that define core matrix transformations
    '''
    def translate(self, shape, Tx, Ty):
        '''
        Translate a shape according to translation parameters
        '''
        return Shape([shape.x + Tx, shape.y + Ty])

    def normalize(self, shape):
        '''
        Perform isomorphic scaling in order to normalize shapes
        See Amy Ross on GPA p.5

        in: target Shape
        out: scaled Shape object
        '''
        return Shape([shape.vector/np.linalg.norm(shape.vector)])

    def scale_and_rotate(self, subject, s, theta, inverse=False):
        '''Rotate over theta and scale by s'''
        rotation_matrix = np.array([
                            [s*math.cos(theta), -1*s*math.sin(theta)],
                            [s*math.sin(theta), s*math.cos(theta)]
                            ])
        if inverse:
            return Shape(np.dot(rotation_matrix.T, subject.matrix))
        else:
            return Shape(np.dot(rotation_matrix, subject.matrix))

    def transform(self, target, pose_parameters):
        '''
        Perform transformations to move target.

        in: Shape target,
            translation params Tx, Ty,
            scaling param s,
            rotation angle theta
        out: transformed Shape
        '''
        # normalize target to avoid multiple scaling
        target = self.normalize(target)
        # get pose parameters and perform transformations
        Tx, Ty, s, theta = pose_parameters
        scaled_and_rotated = self.scale_and_rotate(target, s, theta)
        return self.translate(scaled_and_rotated, Tx, Ty)

    def invert_transform(self, target, pose_parameters):
        '''
        Computes an inverse transformation of target
        '''
        # get pose parameters and invert transform
        Tx, Ty, s, theta = pose_parameters
        translated = self.translate(target, -1*Tx, -1*Ty)
        return self.scale_and_rotate(translated, 1/s, theta, inverse=True)


class CoreFinder(object):
    '''
    Methods to get the correct pose parameters of a shape with
    respect to a target shape. For scaling and rotation, first
    get the transformation parameters a and b.
    '''
    def get_transformation_parameters(self, subject, target):
        '''
        Compute the parameters for scaling and rotation.
        As in Cootes (2000), if:
            a = (x1*x2)/|x1|^2
            b = sum1_n(x1i*y2i - y1i*x2i)/|x1|^2
        then
            s^2 = a^2 + b^2
            theta = tan^-1(b/a)
        '''
        # compute denominator
        denom = np.linalg.norm(subject.vector)**2
        # compute numerators
        num_a = np.dot(subject.vector, target.vector.T)
        num_b = np.sum(subject.x.dot(target.y) - subject.y.dot(target.x))
        return num_a/denom, num_b/denom

    def find_scale(self, a, b):
        return math.sqrt(a**2 + b**2)

    def find_rotation_angle(self, a, b):
        '''Return angle'''
        return math.atan(b/a)

    def find_translation(self, subject, target):
        '''
        Translation params = the difference between the centroids
        '''
        scX, scY = subject.centroid()
        tcX, tcY = target.centroid()
        return (tcX - scX, tcY - scY)


class Aligner(CoreAlign, CoreFinder):
    '''
    Alignment class that combines the alignment methods from CoreAlign and the
    transformation variable finding methods from CoreFinder.
    '''
    def get_pose_parameters(self, subject, target):
        '''
        Find the pose parameters to align subject with target.
        In: 1xC array shape1, shape2
        Out: X_t, Y_t params that define the translation,
            s, defines the scaling
            theta, defines the rotation
        '''
        # find/perform transformations
        Tx, Ty = self.find_translation(subject, target)
        # compute a and b
        a, b = self.get_transformation_parameters(subject, target)
        s = self.find_scale(a, b)
        theta = self.find_rotation_angle(a, b)

        return (Tx, Ty, s, theta)
