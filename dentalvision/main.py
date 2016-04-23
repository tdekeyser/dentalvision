import numpy as np
import matplotlib.pyplot as plt

from pdm.model import create_pdm
from asm.model import ActiveShapeModel
from asm.fit import Aligner, fit
from alignment.shape import Shape


def run():
    # Create deformable model based on landmark data
    paths = [
        '../Project Data/_Data/Landmarks/original/',
        '../Project Data/_Data/Landmarks/mirrored/'
        ]
    pdmodel, landmarks = create_pdm(paths)

    # ############TESTS
    target_landmark = landmarks[0]
    mean = pdmodel.mean
    target = Shape(target_landmark)

    # test fit function
    Tx, Ty, s, theta, c = fit(pdmodel, target_landmark)

    asm = ActiveShapeModel(pdmodel)
    targy = asm.transform(Tx, Ty, 1, theta, c)
    aligned_mean = asm.transform(Tx, Ty, s, theta, np.zeros(pdmodel.dimension))
    print targy.centroid()

    aligner = Aligner()
    variation = pdmodel.deform(c)

    plt.plot(target.x, target.y, color='k', marker='o', label='target')
    plt.plot(targy.x, targy.y, color='r', marker='o', label='result')

    plt.scatter(aligned_mean.x, aligned_mean.y, color='y', marker='o', label='aligned_mean')

    # plt.plot(mean.x, mean.y, color='g', marker='o', label='mean')
    # plt.plot(variation.x, variation.y, color='r', marker='o', label='model variation')

    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    run()
