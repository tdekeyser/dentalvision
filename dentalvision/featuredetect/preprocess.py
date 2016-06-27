'''
Preprocessing step for creating a training set for feature detector.

Imagemagick command to resize the images to EXACTLY the same size:
    $ convert $i -resize 110x320^ -gravity center -extent 110x320 $i
    $ convert $i -flip $i
'''
import cv2
from utils.structure import Shape


def slice_images(images, landmarks_per_image):
    '''
    Slice the landmarks from the images
    '''
    imagecount, shapecount, landmarkcount = landmarks_per_image.shape
    for i in range(imagecount):
        image = images[i]
        for s in range(shapecount):
            landmark = Shape(landmarks_per_image[i, s, :])
            # slice the landmarks from the images
            landmark_image = image[min(landmark.y)-5:max(landmark.y)+5, min(landmark.x)-5:max(landmark.x)+5]
            cv2.imwrite('../Project Data/_Data/Slicings2/' + str(i) + '-' + str(s) + '.png', landmark_image)
