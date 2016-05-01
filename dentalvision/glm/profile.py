'''
Build a grey-level profile length 2k+1 along the normal to a point.

'''
import math
import numpy as np


class Profile(object):
    '''
    Class that creates a normal to input points and computes the 2k nearest
    pixels to that normal.
    '''
    def __init__(self, k):
        self.k = k

    def sample(self, points):
        '''
        Get a sample from self.k points on each side of the normal between
        the triple in points.

        in: triple previous point, point, and next point
        out: list of tuples along the normal through point
        '''
        prev_point, point, next_point = points
        # compute the normal
        self.normal = self._compute_normal(prev_point, next_point)
        # sample along the normal
        return self._sample(point)

    def profile(self, image, points):
        '''
        Compute the distance to normal for each pixel in frame. Return the
        greyscale intensity of the 2k+1 nearest pixels to normal.

        out: list of grey-levels
        '''
        greys = np.asarray([float(image[r, c]) for r, c in self.sample(points)])
        return self._normalize(self._derive(greys))

    def _sample(self, starting_point):
        '''
        Returns 2k+1 points along the normal
        '''
        positives = []
        negatives = []
        start = [(int(starting_point[0]), int(starting_point[1]))]

        i = 1
        while len(positives) < self.k:
            new = (starting_point[0] - i*self.normal[0], starting_point[1] - i*self.normal[1])
            if (new not in positives) and (new not in start):
                positives.append(new)
            i += 1

        i = 1
        while len(negatives) < self.k:
            new = (starting_point[0] + i*self.normal[0], starting_point[1] + i*self.normal[1])
            if (new not in negatives) and (new not in start):
                negatives.append(new)
            i += 1

        negatives.reverse()

        return negatives + start + positives

    def _compute_normal(self, a, b):
        '''
        Compute the normal between two points a and b.

        in: tuple coordinates a and b
        out: 1x2 array normal
        '''
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        tx, ty = (dx/math.sqrt(dx**2+dy**2), dy/math.sqrt(dx**2+dy**2))
        return (-1*ty, tx)

    def _derive(self, profile):
        '''
        Get derivative profile by computing the discrete difference.
        See Hamarneh p.13.
        '''
        return np.diff(profile)

    def _normalize(self, vector):
        '''
        Normalize a vector such that its sum is equal to 1.
        '''
        div = np.sum(vector) if bool(np.sum(vector)) else 1
        return vector/div
