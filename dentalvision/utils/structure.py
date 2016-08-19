'''
Basic class for shapes. Provides easy access to array, matrix, and coordinate
arrays. Input should be 1xC array OR 2-element list of arrays:

>> x = (x1, ..., xC, y1, ..., yC).
>> s = Shape(x)
>> s.x, s.y
array([x1, ..., xC]), array([y1, ..., yC])
>> s.matrix
array(
    [x1, ..., xC],
    [y1, ..., yC]
    )
>> s.array
array([x1, ..., xC, y1, ..., yC])
>> s.centroid()
(centroid_x, centroid_y)
'''
import numpy as np


class Shape(object):
    '''
    Class that represents a basic shape.
    In: array in form (x1, ..., xC, y1, ..., yC) OR
        list [array([x1, ..., xC]), array([y1, ..., yC])]
    '''
    def __init__(self, array):
        try:
            self.x, self.y = np.split(np.hstack(array), 2)
        except AttributeError:
            self.x, self.y = array
        self.length = self.x.size
        self.vector = np.hstack((self.x, self.y))
        self.matrix = np.vstack((self.x, self.y))

    def __sub__(self, other):
        '''
        Defines subtraction of a Shape with another Shape.
        '''
        return Shape([self.x - other.x, self.y - other.y])

    def __add__(self, other):
        '''
        Defines addition of a Shape with another Shape.
        '''
        return Shape([self.x + other.x, self.y + other.y])

    def __mul__(self, m):
        '''
        Defines multiplication of a Shape with int scalar.
        '''
        return Shape(self.vector * m)

    def __div__(self, d):
        '''
        Defines division of a Shape with int scalar.
        '''
        return Shape(self.vector / d)

    def get(self, index):
        '''
        Get coordinate based on index.
        '''
        return np.array([self.x[index], self.y[index]])

    def centroid(self):
        '''
        Compute the centroid: the average of an array of coordinates.
        '''
        return (np.sum(self.x)/self.x.shape, np.sum(self.y)/self.y.shape)

    def slice(self, split_index):
        '''
        Split the Shape with steps of split_index.
        '''
        return Shape((self.x[::split_index], self.y[::split_index]))       

    def merge(self, other):
        return Shape(np.hstack((np.hstack((self.x, other.x)), np.hstack((self.y, other.y)))))
