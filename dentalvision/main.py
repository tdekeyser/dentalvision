import numpy as np
import cv2

from pdm.model import create_pdm
from glm.model import create_glm
from asm.model import ActiveShapeModel
from featuredetect.model import create_featuredetectionmodel
from utils.multiresolution import gaussian_pyramid
from utils.structure import Shape
from utils import plot
from loader import DataLoader


MATCH_DIM = (320, 110)          # dimensions searched by feature detector
LANDMARK_AMOUNT = 40            # amount of landmarks per tooth


def run():
    '''
    Main method of the package.

    1. Loading:
        Loads data from the input constants in loader.py.
        Returns training and test sets from the image files and
        the landmark files.
        The image data sets is then blurred with a median filter
        to remove some noise of the radiographs.

    2. Model SETUP:
        Two systems are trained. For initialisation, a feature
        detector is used that can switch between semi-automatic
        to fully automated search. Semi-automated search involved
        a search in a restricted space. Fully automated search
        uses a trained approximation of a restricted search space.
        Then, an Active Shape Model is trained using the loaded
        radiograph data. The model creates a shape model and a model
        of the gray level profiles around each model point.

    3. Test environment:
        The models are tested by first scanning the image for
        matching regions using the feature detector and then by
        initialising the active shape model on the detected region.
    '''
    # ------------- LOAD DATA -------------- #
    loader = DataLoader()
    trainim, testim, trainlandmarks, testlandmarks, landmarks_per_image = loader.leave_one_out(test_index=4)
    # remove some noise from the image data
    for i in range(trainim.shape[0]):
        trainim[i] = remove_noise(trainim[i])

    # --------------- SETUP ---------------- #
    # train a Feature Detection system
    featuredetector = FDSetup()
    # fully automatic:
    featuredetector.search_region = featuredetector.scan_region(trainlandmarks, diff=20, searchStep=20)
    # semi-automatic:
    # featuredetector.search_region = ((880, 1130), (1350, 1670), 15)     # for radiograph 1
    print '---Search space set to', featuredetector.search_region

    # build and train an Active Shape Model
    asm = ASMSetup(trainim, landmarks_per_image, trainlandmarks, k=10, levels=4)

    # --------------- TEST ----------------- #
    # remove some noise from the test image
    testim = remove_noise(testim)

    # perform feature matching to find init regions
    print '---Searching for matches...'
    matches = featuredetector.match(testim)
    print 'Done.'

    for i in range(len(matches)):
        # plot.render_image(testim, init)
        # search and fit image
        new_fit = asm.activeshape.multi_resolution_search(testim, matches[i], t=20, max_level=3, max_iter=20, n=None)
        mse = np.zeros((1, testlandmarks.shape[0]))
        for i in range(testlandmarks.shape[0]):
            mse[0, i] = mean_squared_error(testlandmarks[i], new_fit)
        min_index = np.argmin(mse)
        print 'MSE:', mse[0, min_index]
        # plot result
        plot.render_image(testim, new_fit, title='result new fit from main.py')
        # plot target
        plot.render_image(testim, testlandmarks[min_index])


def remove_noise(img):
    '''
    Blur image to partially remove noise. Uses a median filter.
    '''
    return cv2.medianBlur(img, 5)


def mean_squared_error(landmark, fit):
    '''
    Compute the mean squared error of a fitted shape w.r.t. a
    test landmark.

    in: np array landmark
        Shape fit
    out: int mse
    '''
    return np.sum((fit.vector - landmark)**2)/fit.length


class FDSetup(object):
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


class ASMSetup(object):
    '''
    Class that creates a complete Active Shape Model.
    The Active Shape Model is initialised by first building a point distribution
    model and then analysing the gray levels around each landmark point.

    in: String image directory
        String landmark directory
    '''
    def __init__(self, images, landmarks_per_image, landmarks, k=8, levels=4):
        self.images = images
        self.landmarks_per_image = landmarks_per_image
        self.landmarks = landmarks

        print '***Setting up Active Shape Model...'
        # 1. POINT DISTRIBUTION MODEL
        print '---Building Point-Distribution Model...'
        self.pdmodel = self.pointdistributionmodel(landmarks)
        print 'Done.'

        # 2. GRAYSCALE MODELs using multi-resolution images
        print '---Building Gray-Level Model pyramid...'
        self.glmodel_pyramid = self.grayscalemodel_pyramid(k=k, levels=levels)
        print 'Done.'

        # 3. ACTIVE SHAPE MODEL
        print '---Initialising Active Shape Model...'
        self.activeshape = ActiveShapeModel(self.pdmodel, self.glmodel_pyramid)
        print 'Done.'

    def pointdistributionmodel(self, landmarks):
        '''
        Create model of shape from input landmarks
        '''
        return create_pdm(landmarks)

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
