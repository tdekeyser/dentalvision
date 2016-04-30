'''
Create a model of the grey level around each landmark. The goal is to produce a
measure in the search for new model points.
In the training stage, a grey level profile vector of length 2k+1 is made for
each landmark. Instead of using actual grey levels, normalised derivatives are
used. The Mahalanobis distance is used as an evaluation measure.
'''
import numpy as np

from glm.profile import Profile


def create_glm(images, shapes, k=0):
    '''
    Create a gray-level model

    in: array of images,
        array of shapes,
        amount of pixels k examined on either side
            of the normal
    out: grayscale level model
    '''
    glmodel = GrayLevelModel(0, k)

    # get gray-level profiles for each landmark
    landmark_profiles = glmodel._get_image_profiles(images, shapes)
    landmark_amount = landmark_profiles.shape[0]
    glmodel.amount_of_landmarks = landmark_amount

    # perform PCA for each landmark
    for l in range(landmark_amount):
        profiles = landmark_profiles[l]
        # build the model using a covariance matrix and mean per landmark
        mean = profiles.mean(0)
        covariance = np.dot(profiles.T, profiles)
        glmodel.build(covariance, mean)
        print covariance.shape

    return glmodel


class GrayLevelModel(object):
    '''
    Build a grey-level model that is able to evaluate the grey-level
    of a new pattern to a mean for that landmark.
    '''
    def __init__(self, amount_of_landmarks, k):
        self.amount_of_landmarks = amount_of_landmarks
        self.covariance = []
        self.mean = []
        self.k = k
        self.profiler = Profile(k)
        self.M_index = 0

    def get(self, index):
        '''
        Get eigen*s and mean from landmark index (int)
        '''
        return self.covariance[index], self.mean[index]

    def build(self, covariance, mean):
        '''Update the model with new landmark eigen*s and mean'''
        self.covariance.append(covariance)
        self.mean.append(mean)

    def set_evaluation_index(self, m):
        self.M_index = m

    def evaluate(self, profile):
        '''
        Evaluate the quality of the fit of a new sample according
        to the Mahalanobis distance:

            f(g_new) = (g_new - mean).T * cov^-1 * (g_new - mean)

        in: vector of grey-level profile
        out: Mahalanobis distance measure
        '''
        cov, mean = self.get(self.M_index)
        # np.linalg.inv(cov) returns Singular Matrix error --> not invertible...
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
        profiles = []
        for s in range(shapes.shape[2]/2):
            profiles.append(np.zeros((images.shape[0], 2*self.k)))

        # iterate over all images, all shapes, and all landmarks to extract grey-level profiles
        for i in range(len(images)):
            # transpose image to be able to place correct coordinates (x, y)
            image = images[i].T
            for j in range(shapes[i].shape[0]):
                shape = np.vstack(np.split(shapes[i][j], 2))
                for l in range(shape.shape[1]):
                    # make grayscale profile for each landmark couple
                    profile = self.profile(image, (shape[:, l-2], shape[:, l-1], shape[:, l]))
                    # normalize and derive profile
                    profiles[l-1][i, :] = profile

        return np.asarray(profiles)

    # def project(self, profile, eigenvector, mean):
    #     '''
    #     Project profile onto one of the eigenvectors.
    #     See Cootes (1993) p.641:
    #         b_g = eigenvectors.T * (g - mean)

    #     out: vector of profile parameters
    #         eigenvalues that go with the profile
    #     '''
    #     return np.dot(eigenvector.T, (profile - mean).T)
