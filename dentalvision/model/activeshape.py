'''
Given initial knowledge of where the target object lies in an image,
the Active Shape Model algorithm amounts to a directed search of the
parameter space (Cootes 2000, 12).

See algorithm in Cootes (2000), p. 12-13.
'''
import numpy as np
from model.deformable import create_deformable_model
from model.match import match


class ActiveShapeModel(object):
    '''
    Algorithm examines a region close to the initial position. For each
    point X_i in this region, find the shape/pose parameters of the
    deformable model that fits the examined region (keeping the shape
    parameter within a 3*sqrt(eigval) bound).
    Repeat until convergence.
    '''
    pass
