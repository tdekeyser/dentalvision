import math
import numpy as np

from pdm.pca import pca


class EigenModel(object):
    '''
    Create an eigen model from the training images in order
    to find initial positions for the Active Shape Model.

    in: np array of training images, for which each row is an image
    '''
    def __init__(self, training_images):
        # build the eigen model
        self._build(training_images)

    def project_and_reconstruct(self, img):
        '''
        Project an image on its eigenvectors, then reconstruct the
        image using its principal components.
        '''
        return self.reconstruct(self.project(img))

    def project(self, img):
        '''
        Project img on the space spanned by the eigenvectors.

        See p.672 (14.11) in Szleski
        '''
        # compute the optimal coefficients a_i for any new image X
        # these are the best approximation coefficients for X
        return (img-self.mean).dot(self.eigenvectors)

    def reconstruct(self, projection):
        '''
        Reconstruct an image based on its PCA-coefficients, the eigenvectors
        and the average.

        See Szleski p.671 (14.8)
        '''
        # start with the mean image and add small numbers of scaled signed images
        return self.mean + projection.dot(self.eigenvectors.T)

    def _build(self, images):
        # perform PCA on images
        self.eigenvalues, self.eigenvectors, self.mean = pca(images, max_variance=0.98)
        # # eigenvectors need to be orthogonal and of unit form
        # self.eigenvectors = self._normalize(eigenvectors)

    def _normalize(self, vectors):
        return vectors/math.sqrt(np.sum(vectors**2))
