'''
Extract a grey level profile from a set of landmarks.
'''
import numpy as np
import math


def extract_profile(image, landmarkcouple, k=3):
    landmark1, landmark2 = landmarkcouple
    # create frame with radius k
    x, y = landmark1
    frame = image[x-k:x+k, y-k:y+k]
    # create normal to landmarks
    profilemaker = Profile(landmark1, landmark2)
    # get 2k points closest to frame
    profile = profilemaker.get_profile(frame, k)
    profile.insert(k-1, image[x, y])
    return np.array(profile)


class Profile(object):
    '''
    Class that creates a normal to input points and computes the 2k nearest
    pixels to that normal.
    '''
    def __init__(self, a, b):
        self.perpendicular(a, b)        

    def get_profile(self, frame, k):
        '''
        Compute the distance to normal for each pixel in frame. Return the
        grayscale intensity of the 2k nearest pixels to normal.
        '''
        rows, columns = frame.shape
        distances = []
        for r in range(rows):
            for c in range(columns):
                framedist = (self.distance((r, c)), frame[r, c])
                distances.append(framedist)
        return [float(g[1]) for g in sorted(distances, reverse=True)[:2*k]]

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

    def distance(self, point):
        x, y = point
        if self.slope != 0:
            return (-1*self.slope*x + y)/math.sqrt(math.pow(self.slope, 2)+1)
        else:
            return x
