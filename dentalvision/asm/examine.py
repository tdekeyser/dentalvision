import numpy as np

from utils.shape import Shape
from utils.profile import extract_profile, Profile


def examine(image, glmodel, model_points, t=3):
    '''
    Examines points normal to model points and compare its grey-levels
    with the grey-level model.

    in: matrix of pixels image
        GreyLevelModel glmodel
        array of model points (x1, x2, ..., xN, y1, y2,..., yN)
        int t amount of pixels examined either side of the normal (t > k)
    out: array of new points (x1, x2, ..., xN, y1, y2,..., yN)
    '''
    # transpose image for correct placement of x and y
    image = image.T
    model_points = Shape(model_points)
    for m in range(model_points.length):
        # get point and next
        point = model_points.get(m)
        next_point = model_points.get(m-1)
        # slice frame around model point
        x, y = point
        frame = image[x-t:x+t, y-t:y+t]
        # determine t points along the normal
        profile = Profile(point, next_point)
        profile_coordinates = profile.get_profile(frame, t, coordinates=True)
        # determine grey-level profiles of all found points
        for i in range(len(profile_coordinates)):
            coord = profile_coordinates[i]
            next_coord = profile_coordinates[i-1]
            profile = extract_profile(image, (coord, next_coord), k=t)
            # for each point in profile, evaluate
            
