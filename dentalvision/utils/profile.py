'''
Build a grey-level profile length 2k+1 along the normal to a point.

'''
import math
import numpy as np


def extract_profile(image, couple, k=3):
    '''
    Extracts a grey-level profile for a point.

    in: matrix of pixels image
        tuple of 2 tuples coordinates (points)
    out: grey-level profile of the first point
    '''
    point1, point2 = couple
    x, y = point1
    # create frame with radius k
    frame = image[x-k:x+k, y-k:y+k]
    # create normal to landmarks
    profilemaker = Profile(point1, point2)
    # get 2k+1 points closest to frame
    profile = profilemaker.get_profile(frame, k)
    return np.array(profile)


class Profile(object):
    '''
    Class that creates a normal to input points and computes the 2k nearest
    pixels to that normal.
    '''
    def __init__(self, a, b):
        self.normal(a, b)

    def get_profile(self, frame, k):
        '''
        Compute the distance to normal for each pixel in frame. Return the
        greyscale intensity of the 2k+1 nearest pixels to normal.

        out: list of tuples (coordinates, grey-levels)
        '''
        return [frame[r, c] for r, c in self.get_closest(frame, k)]

    def get_closest(self, frame, k):
        '''
        Returns 2k+1 closest points along the normal and within frame
        '''
        rows, columns = frame.shape
        distances = []
        for r in range(rows):
            for c in range(columns):
                framedist = (self.distance((r, c)), (r, c))
                distances.append(framedist)
        return [float(g[1]) for g in sorted(distances, reverse=True)[:2*k+1]]

    def normal(self, a, b):
        '''
        Set the slope of a normal through a on ab as a class variable.

        in: 2 tuples of coordinates
        '''
        x1, y1 = a
        x2, y2 = b
        self.slope = -1*(x2 - x1)/(y2 - y1) if bool(y2-y1) else 0

    def distance(self, point):
        '''
        Compute euclidean distance from input point to the normal
        of the profile using its slope.

        in: tuple point
        out: int distance
        '''
        x, y = point
        return (-1*self.slope*x + y)/math.sqrt(math.pow(self.slope, 2)+1) if bool(self.slope) else x
