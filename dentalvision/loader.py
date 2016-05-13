import os
import cv2
import numpy as np


IMAGE_DIR = '../Project Data/_Data/Radiographs/'
IMAGE_AMOUNT = 14
IMAGE_DIM = (3023, 1597)

LANDMARK_DIR = '../Project Data/_Data/Landmarks/original/'
LANDMARK_AMOUNT = 40            # amount of landmarks per tooth


class DataLoader(object):
    '''
    Loads images and landmarks, both as array and as list of arrays per image.
    '''
    def __init__(self):
        self.image_list = os.listdir(IMAGE_DIR)
        self.landmark_list = os.listdir(LANDMARK_DIR)
        self._load_datasets()

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

        training_landmarks_per_image = np.vstack((self.landmarks_per_image[:test_index], self.landmarks_per_image[test_index+1:]))

        # create landmark training and test sets
        training_landmarks = np.vstack(training_landmarks_per_image[:][:])
        test_landmarks = np.vstack(self.landmarks_per_image[test_index][:])

        return training_images, test_images, training_landmarks, test_landmarks, training_landmarks_per_image

    def _load_datasets(self):
        '''
        Load the image and landmark dataset.
        '''
        # compile images
        self.images = []
        for i in range(len(self.image_list)):
            if self.image_list[i].endswith('.tif'):
                path = IMAGE_DIR + self.image_list[i]
                self.images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))

        # compile landmarks per image for convenience in grayscale level training
        self.landmarks_per_image = []
        for i in range(len(self.images)):
            self.landmarks_per_image.append([self._load(LANDMARK_DIR + s) for s in self.landmark_list if 'landmarks'+str(i+1)+'-' in s])
        self.landmarks_per_image = np.asarray(self.landmarks_per_image)

    def _load(self, path):
        '''
        Load and parse the data from path, and return arrays of x and y coordinates
        Data should be in the form (x1, y1)

        in: String pathdirectory
        out: 1xc array (x1, ..., xN, y1, ..., yN)
        '''
        data = np.loadtxt(path)
        x = data[::2, ]
        y = data[1::2, ]
        return np.hstack((x, y))
