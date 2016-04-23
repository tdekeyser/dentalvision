'''
Extract a grey level profile from a set of landmarks.
'''
import numpy as np
import math


def extract_profile(image, landmarkcouple, k=3):
    landmark1, landmark2 = landmarkcouple


class PixelGet(object):
    def __init__(self, a, b):
        self.perpendicular(a, b)

    def perpendicular(self, a, b):
        '''
        Create a perpendicular through a on ab
        input = 2 tuples
        '''
        x1, y1 = a
        x2, y2 = b
        try:
            self.slope = -1*(y2 - y1)/(x2 - x1)
        except ZeroDivisionError:
            self.slope = 0
        self.intercept = (self.slope * (-1) * x1) + y1

    def distance(self, point):
        x, y = point
        return (-1 * self.slope * x + y - 1*self.intercept)/math.sqrt(math.pow(self.slope, 2)+1)
        