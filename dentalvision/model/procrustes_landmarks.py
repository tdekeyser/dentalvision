# -*- coding: utf-8 -*-
'''
Preprocess data by performing procrustes analysis on the tooth landmarks.
See the paper by Amy Ross (https://cse.sc.edu/~songwang/CourseProj/proj2004/ross/ross.pdf)
for a summary of procrustes analysis.

Step 1 - translation -translational components can be
    removed from an object by translating the object so that the mean of all
    the object's points (i.e. its centroid) lies at the origin.
Step 2 - scaling
Step 3 - rotate
'''
import numpy as np


def centroidnp(x,y):
    '''
    x = all x-coorodinates/landmarks
    y = all y-coordinates/landmarks
    output = int,int
    '''
    xlength = x.shape[0]
    ylength = y.shape[0]
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    centrx = sum_x/xlength
    centry = sum_y/ylength
    return centrx, centry
    
def translate(x,y,centroidx, centroidy):
    '''
    Output = array, array 
    '''
    translcentroidx = x - centroidx
    translcentroidy = y - centroidy
    return translcentroidx, translcentroidy
		

def scaling(transx,transy):
    '''
    output = array, array
    '''
    scalingx = np.sqrt(np.sum(np.power(transx, 2.0)/float(transx.size)))
    scalingy = np.sqrt(np.sum(np.power(transy, 2.0)/float(transy.size)))
    scaledx = transx/scalingx
    scaledy = transy/scalingy
    return scaledx, scaledy


def rotate(matrixxy,angle):
    '''
    rotate all points according to certain angle
    output = array
    '''
    rotation = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),  np.cos(angle)]])
    return matrixxy.dot(rotation)
 
       
def make_xy_matrix(x,y):
    xymatrix = np.vstack((x,y))
    matrixrm = np.transpose(xymatrix)
    return matrixrm


def procrustes_analysis(infile):
    data = np.loadtxt(infile)
    x = data[::2,]
    y = data[1::2,]
    #step1
    center_x, center_y = centroidnp(x, y)
    #step2
    trans_x, trans_y = translate(x,y,center_x, center_y)
    #step 3 - scale
    scale_x, scale_y = scaling(trans_x, trans_y)
    #step - convert to xy - matrix
    matrix_xy = make_xy_matrix(x,y)
    #step 4 - rotation
    rotated = rotate(matrix_xy,0.5)
    print rotated
    
    
    
    
  	
if __name__ == '__main__':
    f="landmarks8-8.txt"
    procrustes_analysis(f)
