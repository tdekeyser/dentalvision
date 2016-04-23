'''
Create a model of the grey level around each landmark. The goal is to produce a
measure in the search for new model points.
In the training stage, a grey level profile vector of length 2k+1 is made for
each landmark. Instead of using actual grey levels, normalised derivatives are
used.
'''
import numpy as np


def create_greylevelmodel(paths, landmarks, k=3):
    pass


class GreyLevelModel(object):
    def __init__(self, eigenvalues, eigenvectors, mean):
        pass
    
    def evaluate(self, profile):
        '''
        Use the Mahalanobis distance to evaluate the fit of a new profile.
        '''
        pass
