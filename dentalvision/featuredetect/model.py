import os
import cv2
import numpy as np

from utils.eigen import EigenModel


# try flippped/ or nonflipped/
ROOT = '../Project Data/_Data/Slicings/nonflipped/'


def create_featuredetectionmodel():
    # build training set
    training_images = build_training_array()
    # set up the detector instance
    return FeatureDetection(training_images)


def build_training_array():
    '''
    Create array of all training images using the sliced landmark
    shapes. Returns array of images, for which each row is an image.
    '''
    images = np.zeros((112, 35200))
    index = 0
    for i in os.listdir(ROOT):
        image = cv2.cvtColor(cv2.imread(ROOT + i), cv2.COLOR_BGR2GRAY)
        image = cv2.medianBlur(image, 3)

        images[index, :] = np.hstack(image)
        index += 1
    return images


class FeatureDetection(object):
    def __init__(self, training_images):
        # make eigen model from the images
        self.eigenmodel = EigenModel(training_images)

    def match(self, image, search_region, target_dimension):
        '''
        Use eigen model to find shapes in an input image.

        in: np array image
            tup(tup, tup, int) search_region; defines the possible
                region of the shape centroid. Search iteratively proceeds
                this region.
            tup(int, int) target_dimension; dimension of the sliced
                training shapes
        '''
        searchX, searchY, searchStep = search_region
        dimX, dimY = target_dimension

        matches = []
        for search_row in range(2):
            for y in range(searchY[0], searchY[1], searchStep):
                centerX, centerY = searchX[search_row], y
                # slice frame from the image
                frame = image[
                            centerX-(dimX/2):centerX+(dimX/2),
                            centerY-(dimY/2):centerY+(dimY/2)
                        ]

                # evaluate the model
                reconstructed_frame = self.eigenmodel.project_and_reconstruct(np.hstack(frame))
                distance = np.sum(np.absolute(self.eigenmodel.mean - reconstructed_frame))

                matches.append((distance, np.array([centerY, centerX])))
        # return centers
        return [match[1] for match in self._get_distinct(matches)]

    def _get_distinct(self, indistincts):
        '''
        Remove elements that are too close to each other.
        '''
        # sort input
        indistincts = sorted(indistincts)
        # only take distinct matches
        distinct_matches = [indistincts[0]]
        for s in range(1, len(indistincts)):
            if not any(abs(int(np.sum(d[1]) - np.sum(indistincts[s][1]))) < 70 for d in distinct_matches):
                distinct_matches.append(indistincts[s])
        return distinct_matches
