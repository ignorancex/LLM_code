import numpy as np

from gravann.labels import calculate_volume

cube_vertices = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1]
])

cube_faces = np.array([
    [1, 3, 2],
    [0, 3, 1],
    [0, 1, 5],
    [0, 5, 4],
    [0, 7, 3],
    [0, 4, 7],
    [1, 2, 6],
    [1, 6, 5],
    [2, 3, 6],
    [3, 7, 6],
    [4, 5, 6],
    [4, 6, 7]
])

tetraeder_vertices = np.array([
    [-0.5, 0, 0],
    [0.5, 0, 0],
    [0, 0.86602540378443864676372317075, 0],
    [0, 0.28867513459481288225457439025, 0.8164965809277260327324280249019]
])

tetraeder_faces = np.array([
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3]
])


def test_cube_volume():
    assert 8 == calculate_volume(cube_vertices, cube_faces)


def test_tetraeder_volume():
    assert 0.1178511301977579 == calculate_volume(tetraeder_vertices, tetraeder_faces)
