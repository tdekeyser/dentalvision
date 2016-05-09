import numpy as np

from utils.structure import Shape
from utils.align import Aligner


def create_initmodel(landmarks, mean_shape):

    initmodel = InitModel(mean_shape)
    # analyse the images w.r.t. scale, translation, rotation
    initmodel._analyse_training_shapes(landmarks)


class InitModel(object):
    '''
    Model for ASM initialisation. It combines a mean for
    the shape and pose parameters.

    in: Shape mean shape
        int variance; amount of different initiatisation
            possibilities, i.e. 8 in the case of 8 different
            incisors.
    '''
    def __init__(self, shape, variance):
        self.variance = variance
        self.shape = shape
        # define mean pose parameters
        self.scale = np.zeros(1, variance)
        self.translation = np.zeros(2, variance)
        self.rotation = np.zeros(1, variance)
        # initialise aligning class
        self.aligner = Aligner()

    def get_pose(self, i):
        '''
        Get pose parameters according to index.
        '''
        return (self.translation[i], self.scale[i], self.rotation[i])

    def initialise(self, image, i):
        '''
        Returns the mean shape aligned according to the i-th mean
        position.
        '''
        return self.aligner.transform(self.shape, self.get_pose(i))

    def _analyse_training_shapes(self, landmarks):
        '''
        Analyse training images according to their position, scale
        and rotation w.r.t. a mean shape.

        in: np arrays for which the rows are landmark data
        '''
        for l in range(landmarks.shape[0]):
            landmark = Shape(landmarks[l, :])
            Tx, Ty, s, theta = self.aligner.get_pose_parameters(self.mean, landmark)
