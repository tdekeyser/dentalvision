'''
Algorithm for matching the model to image points.

See Cootes (2000) p. 9
'''
import numpy as np
from preprocess.align import AlignmentFinder


def match(deformable_model, image_points):
    '''
    Algorithm that iteratively finds the best shape parameters
    that match the image points.

    In: array of image points (x1, x2, ..., xN, ..., y1, y2,..., yN)
    '''
    # TODO
    # use a DeformableModel that can change the shape parameters
    # use AlignmentFinder to find the pose parameters (Xt, Yt, s, delta)
    return
