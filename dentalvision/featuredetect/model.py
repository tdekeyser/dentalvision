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
        # convert to grayscale and blur to remove some noise
        image = cv2.cvtColor(cv2.imread(ROOT + i), cv2.COLOR_BGR2GRAY)
        image = cv2.medianBlur(image, 3)

        images[index, :] = np.hstack(image)
        index += 1
    return images


class FeatureDetection(object):
    '''
    Training stage takes a rather long time.
    '''
    def __init__(self, training_images):
        # make eigen model from the images
        self.eigenmodel = EigenModel(training_images)
        self.min_threshold, self.max_threshold = self._train_threshold(training_images)

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
        (x_min, x_max), (y_min, y_max), searchStep = search_region

        matches = []
        for x in range(x_min, x_max, searchStep):
            for y in range(y_min, y_max, searchStep):
                # slice frame from the image
                frame = self._slice(image, (x, y), target_dimension)
                # evaluate the frame w.r.t. the model
                evaluation = self._evaluate(frame)
                # add to match if evaluation lies within threshold
                if self.min_threshold <= evaluation <= self.max_threshold:
                    matches.append(np.array([evaluation, y, x]))
        # return centers
        return self._get_distinct(matches)

    def _train_threshold(self, images):
        '''
        Train a threshold by computing the minimum and maximum evaluation
        of the training images.
        '''
        evaluations = np.apply_along_axis(self._evaluate, axis=1, arr=images)
        return (min(evaluations), max(evaluations))

    def _evaluate(self, frame):
        '''
        Evaluate a frame using Mean Squared Error
        '''
        if frame.shape[0] > 1:
            frame = np.hstack(frame)
        reconstructed_frame = self.eigenmodel.project_and_reconstruct(frame)
        # compute distance from mean
        dist = self.eigenmodel.dist_from_mean(reconstructed_frame)
        return np.sqrt(np.sum(dist**2)/dist.size)

    def _slice(self, img, center, dimension):
        '''
        Slice a frame from input image with center (x, y)
        and dimension (width, height).
        '''
        (x, y), (w, h) = center, dimension
        return img[x-(w/2):x+(w/2), y-(h/2):y+(h/2)]

    def _get_distinct(self, indistincts):
        '''
        Remove elements that are too close to each other.
        '''
        # sort input
        indistincts = sorted(indistincts, key=lambda x: x[0])
        # only take distinct matches
        distincts = [indistincts[0][1:]]
        for s in range(1, len(indistincts)):
            if not any(abs(int(np.sum(d) - np.sum(indistincts[s][1:]))) < 80 for d in distincts):
                distincts.append(indistincts[s][1:])
        return np.asarray(distincts, dtype=np.uint32)
