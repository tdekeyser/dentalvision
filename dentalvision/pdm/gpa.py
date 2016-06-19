'''
Preprocess data by performing generalised procrustes (shape) analysis.
See Ross (https://cse.sc.edu/~songwang/CourseProj/proj2004/ross/ross.pdf) and
Stegmann & Gomez (2002, https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_shape_spaces_imm403.pdf)
for a summary of (generalised) procrustes analysis.
'''
import numpy as np
from utils.structure import Shape
from utils.align import CoreAlign


def gpa(shapes):
    '''
    Perform Generalised Procrustes Analysis

    Important: Bookstein (1997) notes that 2 iteration should be enough to
        achive convergence. However, GPA is not guaranteed to converge.
        Increase the tolerated difference to fasten convergence.

    in: matrix of 1xC vectors in form (x1, ..., xC, ..., y1, ..., yC)
    out: approximate mean shape 1xC vector, matrix of aligned shapes w.r.t. ams
    '''
    rows, columns = shapes.shape
    # get a random seed as approximate mean shape
    mean_shape = shapes[0, :]
    aligner = ShapeAligner(mean_shape)
    new_mean = np.zeros_like(mean_shape)

    # compute a more accurate mean shape
    # proceed as long as mean shape is equal to the new mean
    mean_difference = np.sum(mean_shape - new_mean)
    while mean_difference > float(0.000001):
        new_mean = mean_shape
        aligner.set_mean_shape(new_mean)
        for i in range(rows):
            shapes[i, :] = aligner.align(shapes[i, :])
        # calculate the new mean shape
        mean_shape = np.sum(shapes, axis=0)/rows
        # recalculate difference
        mean_difference = np.sum(mean_shape - new_mean)

    return mean_shape, shapes


class ShapeAligner(CoreAlign):
    '''
    Aligns shapes according to a mean shape. Contains method
    to set a mean shape. Main method is the align method that takes a shape
    as input and aligns it to the mean shape according to the
    following steps:

    Step 1 - translate the shape such that its centroid is situated in
        the origin (0,0)
    Step 2 - scale and normalize the image
    Step 3 - rotate the image in order to align it with the mean
    '''
    def __init__(self, mean_shape):
        self.set_mean_shape(mean_shape)

    def set_mean_shape(self, shape):
        self.mean_shape = Shape(shape)

    def align(self, shape):
        '''
        Align shape with the mean shape and return the aligned shape.
        All arrays in form (x1, ..., xC, ..., y1, ..., yC)
        In: 1xC array shape
        Out: 1xC aligned shape
        '''
        # perform aligning
        translated = self.translate_to_origin(Shape(shape))
        scaled = self.normalize(translated)
        aligned = self.rotate_to_target(scaled, self.mean_shape)
        return aligned.vector

    def translate_to_origin(self, shape):
        '''
        Move all shapes to a common center, most likely the origin (0,0)

        In: array x
            array y
        Out = array, array
        '''
        # compute centroid
        centr_x, centr_y = shape.centroid()
        # translate w.r.t. centroid
        return Shape([shape.x - centr_x, shape.y - centr_y])

    def rotate_to_target(self, subject, target):
        '''
        Rotate shape such that it aligns with the target shape

        in: Shapes
        out: rotated Rx2 matrix of subject
        '''
        # perform singular value decomposition (svd) to get U, S, V'
        u, s, v = np.linalg.svd(target.matrix.dot(np.transpose(subject.matrix)))
        # multiply VU' with subject to get the rotated matrix
        vu = np.transpose(v).dot(np.transpose(u))
        return Shape(vu.dot(subject.matrix))
