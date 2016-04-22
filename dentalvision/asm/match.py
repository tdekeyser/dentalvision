'''
Algorithm for matching the model to image points.

See Cootes (2000) p. 9
'''
import numpy as np
from alignment.shape import Shape
from alignment.align import CoreAlign, CoreFinder


def match(pdmodel, image_points):
    '''
    Algorithm that iteratively finds the best shape parameters
    that match the image points.

    In: PointDistributionModel instance pdm,
        array of new image points (x1, x2, ..., xN, y1, y2,..., yN)
            with N==
    Out: the pose and shape parameters of the model
    '''
    # initiate shape vector b to zero + random comparing b_new
    b = np.zeros(pdmodel.dimension).T
    print b.shape
    # initialise pose parameter finder and aligner
    aligner = Aligner()
    image = Shape(image_points)

    # start iteration
    b_new = np.zeros_like(b) + 0.1
    while np.sum(b_new - b) > 0.00000001:
        # GENERATE model points
        b = b_new
        x = Shape(pdmodel.deform(b))
        # ALIGN the model with the image
        # find pose parameters with new image points
        Tx, Ty, s, theta = aligner.get_pose_parameters(x, image)
        # align image with x
        y = aligner.invert_transform(image, Tx, Ty, s, theta)
        # project y into the tangent plane to mean
        projected_y = y.array/np.dot(y.array, pdmodel.mean.T)
        # update model parameters to projected y
        b_new = np.dot(pdmodel.eigenvectors.T, projected_y - pdmodel.mean)

    return (Tx, Ty, s, theta, b_new)

    '''
    Alternative acc. to Blanz p.4 after alignment of image with mean

    ###from line 40 onwards
    # SVD on scaled eigenvectors of the model
    u, w, v = np.linalg.svd(pdmodel.eigenvectors)
    # introduce weight vector
    n = 0.1
    # calculate the shape vector
    c = v.dot(w.diagonal()/(w.diagonal()**2) + n)).dot(u.T).dot(y)
    # calculate resulted model variation
    x = pdmodel.deform(c)

    '''


class Aligner(CoreAlign, CoreFinder):
    '''
    Alignment class that combines the alignment methods from CoreAlign and the
    transformation variable finding methods from CoreFinder.
    '''
    def get_pose_parameters(self, subject, target):
        '''
        Find the pose parameters to align shape1 with shape2.
        In: 1xC array shape1, shape2
        Out: X_t, Y_t params that define the translation,
            s, defines the scaling
            theta, defines the rotation
        '''
        # find/perform transformations
        Tx, Ty = self.find_translation(subject, target)
        # first translate before scaling and rotating
        translated = self.translate(subject, Tx, Ty)
        # compute a and b
        a, b = self.get_transformation_parameters(translated, target)
        s = self.find_scale(a, b)
        theta = self.find_rotation_angle(a, b)

        return (Tx, Ty, s, theta)
