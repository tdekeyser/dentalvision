'''
Create a model of the grey level around each landmark. The goal is to produce a
measure in the search for new model points.
In the training stage, a gray level profile vector of length 2k+1 is made for
each landmark. Instead of using actual grey levels, normalised derivatives are
used. The Mahalanobis distance is used as an evaluation measure.
'''
import numpy as np

from glm.profile import Profiler


def create_glm(images, shapes, k=0):
    '''
    Create a gray-level model

    in: array of images,
        array of shapes,
        amount of pixels k examined on either side
            of the normal
    out: grayscale level model
    '''
    glmodel = GrayLevelModel(k)
    glmodel.build(images, shapes)
    return glmodel


class GrayLevelModel(object):
    '''
    Build a grey-level model that is able to evaluate the grey-level
    of a new pattern to a mean for that landmark.
    '''
    def __init__(self, k):
        self.k = k
        self.m_index = 0
        self.profiler = Profiler(k=k)

    def build(self, images, shapes):
        '''Build a new gray-level model using images and landmarks'''
        landmark_profiles = self._get_image_profiles(images, shapes)
        landmark_count = landmark_profiles.shape[0]

        self.mean = np.zeros((landmark_count, 2*self.k))
        self.covariance = np.zeros((landmark_count, 2*self.k, 2*self.k))

        for l in range(landmark_count):
            profiles = landmark_profiles[l]
            # build the model using a covariance matrix and mean per landmark
            mean = profiles.mean(0)
            deviation = profiles - mean
            self.mean[l] = mean
            self.covariance[l] = np.cov(deviation.T)

    def get(self, index):
        '''
        Get eigen*s and mean from landmark index (int)
        '''
        return self.covariance[index], self.mean[index]

    def set_evaluation_index(self, m):
        '''
        Set index to different landmark in order to compute
        the Mahalanobis distance of the correct landmark.
        '''
        self.m_index = m

    def evaluate(self, profile):
        '''
        Evaluate the quality of the fit of a new sample according
        to the Mahalanobis distance:

            f(g_new) = (g_new - mean).T * cov^-1 * (g_new - mean)

        in: vector of grey-level profile
        out: Mahalanobis distance measure
        '''
        cov, mean = self.get(self.m_index)
        try:
            return (profile - mean).T.dot(np.linalg.inv(cov)).dot(profile-mean)
        except np.linalg.LinAlgError:
            # np.linalg.inv(cov) returns Singular Matrix error --> not invertible
            return (profile - mean).T.dot(cov).dot(profile-mean)

    def profile(self, image, points):
        '''
        Creates gray-level profile for given point in the image.
        '''
        return self.profiler.profile(image, points)

    def _get_image_profiles(self, images, shapes):
        '''
        Create matrix of graylevel profiles along the normal of all
        landmarks and create a matrix of profiles per landmark.
        profiles = [
                np.array((#images, 2k)),
                ... (#landmarks)
                ]

        in: list of matrices with all images
            list of all shapes per image
        out: matrix of profiles per landmark
        '''
        # create a zero matrix for all profiles per landmark
        profiles = np.zeros((shapes.shape[2]/2, images.shape[0], 2*self.k))

        # iterate over all images, all shapes, and all landmarks to extract grey-level profiles
        for i in range(len(images)):
            # transpose image to be able to place correct coordinates (x, y)
            image = images[i].T
            for j in range(shapes[i].shape[0]):
                shape = np.vstack(np.split(shapes[i, j], 2))
                for l in range(shape.shape[1]):
                    # make grayscale profile for each landmark couple
                    profile = self.profile(image, (shape[:, l-2], shape[:, l-1], shape[:, l]))
                    # normalize and derive profile
                    profiles[l-1, i, :] = profile

        return profiles
