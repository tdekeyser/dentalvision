from model.deformable import create_deformable_model


def run():
    # Create deformable model based on landmark data
    paths = [
        '../Project Data/_Data/Landmarks/original/',
        '../Project Data/_Data/Landmarks/mirrored/'
        ]
    model = create_deformable_model(paths)


if __name__ == '__main__':
    run()
