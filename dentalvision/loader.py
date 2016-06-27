import os
import cv2
import numpy as np


IMAGE_DIR = '../Project Data/_Data/Radiographs/r/'
IMAGE_AMOUNT = 14
IMAGE_DIM = (3023, 1597)

LANDMARK_DIR = '../Project Data/_Data/Landmarks/l/'
LANDMARK_AMOUNT = 40            # amount of landmarks per tooth


class DataLoader(object):
    '''
    This class provides methods to load specific landmark datasets
    for training and testing. It loads images and landmarks from
    directory paths specified in constants IMAGE_DIR and LANDMARK_DIR.
    '''
    def __init__(self):
        self.images = self._load_grayscale_images()
        self.landmarks_per_image = self._load_landmarks_per_image()

    def leave_one_out(self, test_index=0):
        '''
        Divides into training and test sets by leaving one image and its
        landmarks out of the training set.

        in: int test_index; index to divide training/test
        out: np array images; array with grayscaled images per row
            np array landmarks; array with all landmarks as rows
            list of np arrays landmarks_per_image; array with rows of landmarks
                for each image
        '''
        training_images = np.asarray(self.images[:test_index] + self.images[test_index+1:])
        test_images = self.images[test_index]

        # create landmark training and test sets
        training_landmarks_per_image = np.vstack((self.landmarks_per_image[:test_index], self.landmarks_per_image[test_index+1:]))

        training_landmarks = np.vstack(training_landmarks_per_image[:][:])
        test_landmarks = np.vstack(self.landmarks_per_image[test_index][:])

        # compile training and test sets
        training_set = [training_images, training_landmarks, training_landmarks_per_image]
        test_set = [test_images, test_landmarks]

        return training_set, test_set

    def _load_grayscale_images(self):
        '''
        Load the images dataset.
        '''
        images = []
        for i in os.listdir(IMAGE_DIR):
            if i.endswith('.tif'):
                path = IMAGE_DIR + i
                images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
        return images

    def _load_landmarks_per_image(self):
        '''
        Compile landmarks per image for convenience in grayscale level
        training. This training phase needs an accurate relation between
        the images and their corresponding landmarks.

        Needs to be run after _load_grayscale_images()!
        '''
        if not self.images:
            raise IOError('Images have not been loaded yet.')

        landmarks_per_image = []
        for i in range(len(self.images)):
            # search for landmarks that include reference to image in path
            lms = [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if 'landmarks'+str(i+1)+'-' in s]
            landmarks_per_image.append(lms)

        return np.asarray(landmarks_per_image)

    def _parse(self, path):
        '''
        Parse the data from path directory and return arrays of x and y coordinates
        Data should be in the form (x1, y1)

        in: String pathdirectory with list of landmarks (x1, y1, ..., xN, yN)
        out: 1xc array (x1, ..., xN, y1, ..., yN)
        '''
        data = np.loadtxt(path)
        x = np.absolute(data[::2, ])
        y = np.absolute(data[1::2, ])
        return np.hstack((x, y))
