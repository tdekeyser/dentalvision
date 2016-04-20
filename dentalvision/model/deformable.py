'''
Create a deformable model by preprocessing landmark data
using GPA and PCA. Based on shape parameters b and eigenvectors P,
the model can construct variations of a mean using the equation:

    x = mean + P*b (see Cootes p.6 eq.2)
    OR
    x = mean + P*diag(eigenvals)*b (see Blanz p.2 eq.7) (=scaled eigenvectors)

Used in the Active Shape Model environment. The Active Shape Model
is then the fit function that iteratively searches for the best
parameters.
'''
import numpy as np


def create_deformable_model(landmark_dirs):
    '''
    Create a new deformable model based on landmark data.

    In: list of directories of the landmark data
    Out: DeformableModel instance created with preprocessed data.
    '''
    # TODO
    # should return a DeformableModel instance
    return


# Would it be useful to create a method for aligning the model with new image points?
# Or simply do the alignment in the Match algorithm?
class DeformableModel(object):
    '''
    Model created based on a mean image and a matrix
    of eigenvectors and corresponding eigenvalues.
    Based on shape parameters, it is able to create a
    variation on the mean shape.
    '''
    def __init__(self):
        # what should be the inputs of the constructor?
        # what should be the class variables?
        # it should have a length (amount of points) for the generation of new image points (must be same)
        # TODO
        pass

    def method1(self):
        # TODO
        # return a variation (=reconstruction)
        return
