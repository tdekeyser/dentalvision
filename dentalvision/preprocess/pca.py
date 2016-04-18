'''
Perform PCA analysis 
'''
import math
import numpy as np


def pca(samples, mean=None, max_variance=1):
    '''
    Perform PCA analysis on samples according to an amount of principal components.

    in: 1xC vector mean of all samples
        RxC matrix containing the samples
        int maximal variance --> limits the amount of returned eigenvectors/values
    out: Rx? matrix of eigenvectors/eigenvalues
        eigenvalues and eigenvectors of the covariance matrix such that their total
        variance is not lower than max_variance.
        1xC vector the average sample
    '''
    n, d = samples.shape

    if not bool(np.sum(mean)):
        mean = np.mean(samples, axis=0)

    # create difference between samples and mean
    deviation_from_mean = samples - mean
    # perform singular value decomposition to get eigenvectors and eigenvalues
    eigenvectors, eigenvalues, variance = np.linalg.svd(deviation_from_mean, full_matrices=False)
    print variance
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    #TODO

    return eigenvalues, normalize_vector(eigenvectors), mean


def normalize_vector(vectors): 
    return vectors/math.sqrt(np.sum(vectors**2))


def build_parameter_model(P, shape, mean):
    '''
    see Cootes p. 6 eq. 3
    '''
    #TODO
    return


def reconstruct(P, b, mean):
    '''
    Reconstruct and image based on principal components and a set of parameters
    that define a deformable model (see Cootes p. 6 eq. 2)

    in: 1xC vector mean shape
        RxT matrix of eigenvectors P
        Tx1 vector deformable model b
    out: ??? reconstructed image
    '''
    #TODO
    return
