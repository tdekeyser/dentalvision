import cv2
import numpy as np


def gaussian_pyramid(image, levels=0):
    '''
    Create a multi-resolutional image pyramid using
    Gaussian smoothing at different levels. Level 0 is
    the original image. Each level is created by first
    smoothing the previous image and then subsampling
    it to get half the amount of pixels of the previous
    image.
    See Cootes et al. (1996) p.14-15.

    Uses OpenCV's pyrDown function to blur and downscale
    and image.

    in: np array with an image
        int amount of levels
    out: list of processed images
    '''
    pyramid = [image]
    for l in range(1, levels):
        pyramid.append(cv2.pyrDown(pyramid[l-1]))
    return np.asarray(pyramid)
