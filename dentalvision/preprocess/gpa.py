'''
Preprocess data by performing generalised procrustes (shape) analysis.
See Ross (https://cse.sc.edu/~songwang/CourseProj/proj2004/ross/ross.pdf) and
Stegmann & Gomez (2002, https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_shape_spaces_imm403.pdf)
for a summary of (generalised) procrustes analysis.

@author: Tina Smets, Tom De Keyser
'''
import numpy as np
from align import ShapeAligner


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
        aligner.set_target_shape(new_mean)
        for i in range(rows):
            shapes[i, :] = aligner.align(shapes[i, :])
        # calculate the new mean shape
        mean_shape = np.sum(shapes, axis=0)/rows
        # recalculate difference
        mean_difference = np.sum(mean_shape - new_mean)

    return mean_shape, shapes
