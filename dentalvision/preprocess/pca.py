'''
Perform PCA analysis 
'''
import math
import numpy as np


def pca(samples, mean, principal_components=0):
    '''
    Perform PCA analysis on samples according to an amount of principal components.

    in: mean of all samples
        np.array containing the samples
        (optional) amount of principal components
    @param principal_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    n, d = samples.shape
    if (principal_components <= 0) or (principal_components > n):
        principal_components = n
    # if not bool(mean):
    #     mean = np.mean(samples, axis=0).reshape(1,-1)

    # create difference between samples and mean
    deviation_from_mean = samples - mean
    # perform singular value decomposition to get eigenvectors and eigenvalues
    eigenvectors, eigenvalues, variance = np.linalg.svd(deviation_from_mean.T, full_matrices=False)
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # select only principal_components
    eigenvalues = eigenvalues[0:principal_components].copy()
    eigenvectors = eigenvectors[:,0:principal_components].copy()
    return eigenvalues, normalize_vector(eigenvectors), mean


def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    
    See p.672 (14.11) in Szleski
    '''
    # compute the optimal coefficients a_i for any new image X
    # these are the best approximation coefficients for X
    return (X-mu).dot(W)


def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    
    See Szleski p.671 (14.8)
    '''
    # start with the mean image and add small numbers of scaled signed images W_i
    return mu + Y.dot(np.transpose(W))


def normalize_vector(vectors): 
    return vectors/math.sqrt(np.sum(vectors**2))


