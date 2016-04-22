import math
import numpy as np

from alignment.shape import Shape


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
        return Shape(
            [shape.x/np.linalg.norm(shape.x), shape.y/np.linalg.norm(shape.y)]
            )

    def scale_and_rotate(self, subject, s, theta, inverse=False):
        '''Rotate over theta and scale by s'''
        rotation_matrix = np.array([
                            [s*math.cos(theta), -1*s*math.sin(theta)],
                            [s*math.sin(theta), s*math.cos(theta)]
                            ])
        if not inverse:
            return Shape(np.dot(rotation_matrix, subject.matrix))
        else:
            return Shape(np.dot(rotation_matrix.T, subject.matrix))

    def transform(self, target, Tx, Ty, s, theta):
        '''
        Perform transformations to move target

        in: Shape target,
            translation params Tx, Ty,
            scaling param s,
            rotation angle theta
        out: transformed Shape
        '''
        scaled_and_rotated = self.scale_and_rotate(target, s, theta)
        return self.translate(scaled_and_rotated, Tx, Ty)

    def invert_transform(self, target, Tx, Ty, s, theta):
        '''
        Computes an inverse transformation of target
        '''
        scaled_and_rotated = self.scale_and_rotate(target, s, theta, inverse=True)
        return self.translate(scaled_and_rotated, -1*Tx, -1*Ty)


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
        denom = np.linalg.norm(subject.array)**2
        # compute numerators
        num_a = np.dot(subject.array, target.array.T)
        num_b = np.sum(subject.x.dot(target.y) - subject.y.dot(target.x))
        return num_a/denom, num_b/denom

    def find_scale(self, a, b):
        return math.sqrt(a**2 + b**2)

    def find_rotation_angle(self, a, b):
        '''Return angle in degrees'''
        return math.atan(b/a)

    def find_translation(self, subject, target):
        '''
        Translation params = the difference between the centroids
        '''
        scX, scY = subject.centroid()
        tcX, tcY = target.centroid()
        return (tcX - scX, tcY - scY)
