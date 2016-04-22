import matplotlib.pyplot as plt

from pdm.model import create_pdm
from asm.model import ActiveShapeModel
from asm.match import Aligner, match
from alignment.shape import Shape


def run():
    # Create deformable model based on landmark data
    paths = [
        '../Project Data/_Data/Landmarks/original/',
        '../Project Data/_Data/Landmarks/mirrored/'
        ]
    model, landmarks = create_pdm(paths)

    #############TESTS

    mean = Shape(model.mean)
    targ = Shape(landmarks[0])

    # aligner = Aligner()
    # Tx, Ty, s, theta = aligner.get_pose_parameters(subj, targ)

    Tx, Ty, s, theta, b = match(model, landmarks[0])
    print b

    asm = ActiveShapeModel(model)
    transformed = asm.transform(Tx, Ty, s, theta, b)

    plt.plot(mean.x, mean.y, color='g', marker='o')
    plt.plot(targ.x, targ.y, color='r', marker='o')
    plt.plot(transformed.x, transformed.y, color='b', marker='o')
    plt.show()


if __name__ == '__main__':
    run()
