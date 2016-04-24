'''
Create a model of the grey level around each landmark. The goal is to produce a
measure in the search for new model points.
In the training stage, a grey level profile vector of length 2k+1 is made for
each landmark. Instead of using actual grey levels, normalised derivatives are
used.
'''
import numpy as np

from glm.profile import extract_profile


def create_glm(images, shapes, k=5):
    '''
    Create a gray level model
    '''
    for i in range(len(images)):
        image = images[i].T
        print image.shape
        imageshapes = shapes[i]
        for shape in imageshapes:
            shape = np.vstack(np.split(shape, 2))
            print shape
            for j in range(shape.shape[1]-1):
                print (shape[:,j], shape[:,j+1])
                # make grayscale profile for each landmark couple
                profile = extract_profile(image, (shape[:,j], shape[:,j+1]), k=k)
                if j == shape.shape[1]-2:
                    # also create a profile of the final landmark
                    profile = extract_profile(image, (shape[:,j+1], shape[:,j]), k=k)
                # normalize profile
                profile = normalize(profile)
                print profile


def normalize(vector):
    return vector/np.sum(vector)


class GreyLevelModel(object):
    def __init__(self, eigenvalues, eigenvectors, mean):
        pass
    
    def evaluate(self, profile):
        '''
        Use the Mahalanobis distance to evaluate the fit of a new profile.
        '''
        pass
