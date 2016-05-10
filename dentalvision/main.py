import os
import cv2
import numpy as np

from pdm.model import create_pdm
from glm.model import create_glm
from asm.model import ActiveShapeModel

from featuredetect.model import create_featuredetectionmodel

from utils.multiresolution import gaussian_pyramid
from utils.structure import Shape
from utils import plot


IMAGE_DIR = '../Project Data/_Data/Radiographs/'
LANDMARK_DIR = '../Project Data/_Data/Landmarks/original/'
LANDMARK_AMOUNT = 40            # amount of landmarks per tooth
MATCH_DIM = (320, 110)          # dimensions searched by feature detector


def run():
    # load images and landmarks
    images, landmarks_per_image, landmarks = load_data()
    # remove some noise from the image data
    images = np.asarray([cv2.medianBlur(image, 5) for image in images])

    # --------------- SETUP ---------------- #
    # set initialisation
    featuredetector = FeatureDetect()
    # semi-automatic:
    # featuredetector.search_region = ((880, 1130), (1350, 1670), 15)     # for radiograph 1
    # fully automatic:
    featuredetector.search_region = featuredetector.scan_region(landmarks, diff=10, searchStep=30)
    print '---Search space set to', featuredetector.search_region

    # build active shape model
    asm = ActiveShapeCreator(images, landmarks_per_image, landmarks)

    # --------------- TEST ----------------- #
    image = cv2.cvtColor(cv2.imread('../Project Data/_Data/Radiographs/13.tif'), cv2.COLOR_BGR2GRAY)
    # remove some noise
    image = cv2.medianBlur(image, 3)

    # perform feature matching to find init regions
    print '---Searching for matches...'
    initial_regions = featuredetector.match(image)
    print 'Done.'

    # image = cv2.Canny(image, 35, 40)

    for init in initial_regions:
        # plot.render_image(image, init)
        # search and fit image
        new_fit = asm.activeshape.multi_resolution_search(image, init, t=20, max_level=4, max_iter=10, n=None)
        # plot result
        plot.render_image(image, new_fit, title='result new fit from main.py')


def load(path):
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


def load_data():
    '''
    Loads images and landmarks, both as array and as list of arrays per image.
    Horrible one-liners, but they do the job.

    out: np array images; array with grayscaled images per row
        list of np arrays landmarks_per_image; array with rows of landmarks for
            each image
        np array landmarks; array with all landmarks as rows, randomly ordered
    '''
    images = np.asarray([cv2.cvtColor(cv2.imread(IMAGE_DIR + im), cv2.COLOR_BGR2GRAY) for im in os.listdir(IMAGE_DIR) if im.endswith('.tif')])
    # images = images[:13]
    landmarks_per_image = []
    for i in range(len(images)):
        landmarks_per_image.append([load(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if 'landmarks'+str(i+1)+'-' in s])
    landmarks = np.asarray([load(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if s.endswith('.txt')])
    return images, landmarks_per_image, landmarks


class FeatureDetect(object):
    '''
    Class that trains a feature detecting system based on an eigen
    model of incisors. Also computes a good search region.
    '''
    def __init__(self):
        print '***Setting up Feature Detector...'
        print '---Training...'
        self.detector = create_featuredetectionmodel()
        print 'Done.'

    def scan_region(self, landmarks, diff=0, searchStep=30):
        '''
        Scan landmark centroids to find a good search region for the feature
        detector.

        in: np array of landmarks Shapes
            int diff; narrows down the search space
            int seachStep; a larger search step fastens search, but also
                increases the risk of missing matches.
        '''
        centroids = np.zeros((landmarks.shape[0], 2))
        for l in range(landmarks.shape[0]):
            centroids[l] = Shape(landmarks[l]).centroid()

        x = (int(min(centroids[:, 0])) + diff, int(max(centroids[:, 0])) - diff)
        y = (int(min(centroids[:, 1])) + diff, int(max(centroids[:, 1])) - diff)

        return (y, x, searchStep)

    def match(self, image, match_frame=MATCH_DIM):
        '''
        Perform feature matching on image in the defined search region.
        Uses the specified target dimension as match region.

        Returns LANDMARK_AMOUNT points along the ellipse of each match. These
        points facilitate alignment with the ASM mean model.

        in: np array image
            tup(tup(int x_min, int x_max), tup(int y_min, int y_max), int searchStep)
                search region; defines the boundaries of the search
            tup match_frame; defines the size of the frame to be sliced
                for matching with the target.
        out: list of np arrays with LANDMARK_AMOUNT points along the ellipse
                around the center of each match.
        '''
        # detect shape with eigen model
        return [self._ellipse(m) for m in self.detector.match(image, self.search_region, match_frame)]

    def _ellipse(self, center, amount_of_points=LANDMARK_AMOUNT):
        '''
        Returns points along the ellipse around a center.
        '''
        print center
        ellipse = cv2.ellipse2Poly(tuple(center), (125, 80), 90, 0, 360, 9)
        return Shape(np.hstack(ellipse[:amount_of_points, :].T))


class ActiveShapeCreator(object):
    '''
    Class that creates a complete Active Shape Model.
    The Active Shape Model is initialised by first building a point distribution
    model and then analysing the gray levels around each landmark point.

    in: String image directory
        String landmark directory
    '''
    def __init__(self, images, landmarks_per_image, landmarks):
        self.images = images
        self.landmarks_per_image = landmarks_per_image
        self.landmarks = landmarks

        print '***Setting up Active Shape Model...'
        # 1. POINT DISTRIBUTION MODEL
        print '---Building Point-Distribution Model...'
        self.pdmodel = self.pointdistributionmodel()
        print 'Done.'

        # 2. GRAYSCALE MODELs using multi-resolution images
        print '---Building Gray-Level Model pyramid...'
        self.glmodel_pyramid = self.grayscalemodel_pyramid(k=8, levels=5)
        print 'Done.'

        # 3. ACTIVE SHAPE MODEL
        print '---Initialising Active Shape Model...'
        self.activeshape = ActiveShapeModel(self.pdmodel, self.glmodel_pyramid)
        print 'Done.'

    def pointdistributionmodel(self):
        '''
        Create model of shape from input landmarks
        '''
        return create_pdm(self.landmarks)

    def grayscalemodel(self, images, k=0, reduction_factor=1):
        '''
        Create a model of the local gray levels throughout the images.

        in: list of np array; images
            int k; amount of pixels examined on each side of the normal
            int reduction_factor; the change factor of the shape coordinates
        out: GrayLevelModel instance
        '''
        return create_glm(images, np.asarray(self.landmarks_per_image)/reduction_factor, k=k)

    def grayscalemodel_pyramid(self, levels=0, k=0):
        '''
        Create grayscale models for different levels of subsampled images.
        Each subsampling is done after by removing half the pixels along
        the width and height of each image.

        in: int levels amount of levels in the pyramid
            int k amount of pixels examined on each side of the normal
        out: list of graylevel models
        '''
        # create Gaussian pyramids for each image
        multi_res = np.asarray([gaussian_pyramid(self.images[i], levels=levels) for i in range(self.images.shape[0])])
        # create list of gray-level models
        glmodels = []
        for l in range(levels):
            glmodels.append(self.grayscalemodel(multi_res[:, l], k=k, reduction_factor=2**l))
            print '---Created gray-level model of level ' + str(l)
        return glmodels


if __name__ == '__main__':
    run()
