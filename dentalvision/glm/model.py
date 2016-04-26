'''
Create a model of the grey level around each landmark. The goal is to produce a
measure in the search for new model points.
In the training stage, a grey level profile vector of length 2k+1 is made for
each landmark. Instead of using actual grey levels, normalised derivatives are
used. The Mahalanobis distance is used as an evaluation measure.
'''
import numpy as np

from utils.profile import extract_profile
from utils.pca import pca


def create_glm(images, shapes, k=3):
    '''
    Create a gray-level model

    in: array of images,
        array of shapes,
        amount of pixels k examined on either side
            of the normal
    out: grayscale level model
    '''
    # get gray-level profiles for each landmark
    landmark_profiles = get_image_profiles(images, shapes, k=k)

    # initialise the model with the amount of landmarks
    glmodel = GreyLevelModel(landmark_profiles.shape[0])

    # perform PCA for each landmark
    for l in range(landmark_profiles.shape[0]):
        profiles = landmark_profiles[l]
        mean = profiles.mean(0)
        # keep all dimensions
        eigenvalues, eigenvectors, m = pca(profiles, mean=mean)
        # build the model using eigen*s and a mean per landmark
        glmodel.build(eigenvalues, eigenvectors, mean)

    return glmodel


def get_image_profiles(images, shapes, k=5):
    '''
    Create matrix of graylevel profiles along the normal of all
    landmarks and create a matrix of profiles per landmark.
    profiles = [
            np.array((#images, 2k)),
            ... (#landmarks)
            ]

    in: list of matrices with all images
        list of all shapes per image
        amount of pixels k examined on either side
            of the normal of each landmark
    out: matrix of profiles per landmark
    '''
    # create a zero matrix for all profiles per landmark
    profiles = []
    for s in range(shapes.shape[2]/2):
        profiles.append(np.zeros((images.shape[0], 2*k)))

    # iterate over all images, all shapes, and all landmarks to
    # extract grey-level profiles
    for i in range(len(images)):
        # transpose image to be able to place correct coordinates (x, y)
        image = images[i].T
        for j in range(shapes[i].shape[0]):
            shape = np.vstack(np.split(shapes[i][j], 2))
            for l in range(shape.shape[1]):
                # make grayscale profile for each landmark couple
                profile = extract_profile(image, (shape[:, l], shape[:, l-1]), k=k)
                # normalize and derive profile
                profiles[l][i, :] = normalize(derivative(profile))

    return np.asarray(profiles)


def derivative(profile):
    '''
    Get derivative profile by computing the discrete difference.
    See Hamarneh p.13.
    '''
    return np.diff(profile)


def normalize(vector):
    '''
    Normalize a vector such that its sum is equal to 1.
    '''
    return vector/np.sum(np.absolute(vector))


class GreyLevelModel(object):
    '''
    Build a grey-level model that is able to evaluate the grey-level
    of a new pattern to a mean for that landmark.
    '''
    def __init__(self, amount_of_landmarks):
        self.amount_of_landmarks = amount_of_landmarks
        self.eigenvalues = []
        self.eigenvectors = []
        self.mean = []

    def get(self, index):
        '''
        Get eigen*s and mean from landmark index (int)
        '''
        return self.eigenvalues[index], self.eigenvectors[index], self.mean[index]

    def build(self, eigenvalues, eigenvectors, mean):
        '''Update the model with new landmark eigen*s and mean'''
        self.eigenvalues.append(eigenvalues)
        self.eigenvectors.append(eigenvectors)
        self.mean.append(mean)

    def evaluate(self, index, profile):
        '''
        Use the Mahalanobis distance to evaluate the quality of
        an input profile.
        See Behiels et al. (1999):

            M = sum_j..n_p(b_gj/eigenvalue_j)

        with b_gj the projection of the input pattern onto the j-th
        eigenvector.

        in: index of landmark
            vector of grey-level profile
        out: Mahalanobis distance measure
        '''
        eigenvalues, eigenvectors, mean = self.get(index)
        M = 0
        for j in range(profile.size):
            # project profile onto profile.size eigenvectors
            project = self.project(profile, eigenvectors[:, j], mean)**2
            M += project/eigenvalues[j]
        return M

    # def project(self, profile, eigenvector, mean):
    #     '''
    #     Project profile onto one of the eigenvectors.
    #     See Cootes (1993) p.641:
    #         b_g = eigenvectors.T * (g - mean)

    #     out: vector of profile parameters
    #         eigenvalues that go with the profile
    #     '''
    #     return np.dot(eigenvector.T, (profile - mean).T)

    # def deform(self, index, profile_param):
    #     '''
    #     Deform a grey-level of a landmark according to a profile parameter b
    #     See Cootes (1993) p.641:
    #         g_new = mean + eigenvectors * b_new

    #     out: deformed profile according to profile parameter
    #     '''
    #     eigenvalues, eigenvectors, mean = self.get(index)
    #     return mean + eigenvectors.dot(profile_param)
