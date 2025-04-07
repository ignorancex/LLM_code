import gym
import numpy as np


class CubeContactMap:
    # only for square: length==width==height
    # four faces: 0 1 2 3
    # default face 0: (0, -w)
    def __init__(self, width, center=0.5):
        self.width = width
        self.center_offset = width * (1 - center) / 2

        self.flat_width = width * center
        # tip_thickness = 0.005
        tip_thickness = 0.00
        self.FaceToCube = [
            lambda x, y: (-width / 2 + x, -width / 2-tip_thickness, -width / 2 + y),
            lambda x, y: (width / 2+tip_thickness, -width / 2 + x, -width / 2 + y),
            lambda x, y: (width / 2 - x, width / 2+tip_thickness, -width / 2 + y),
            lambda x, y: (-width / 2-tip_thickness, width / 2 - x, -width / 2 + y)
        ]

    def compute_face_index(self, x):
        return int(x // self.flat_width)

    def convert_to_cube_space(self, x, y):
        assert 0 <= x <= 4*self.flat_width, f"x should in range (0.0 ~ {self.flat_width})"
        assert 0 <= y <= self.flat_width, f"y should in range (0.0 ~ {self.flat_width})"
        cube_id = self.compute_face_index(x)
        full_flat_x = self.center_offset + x % self.flat_width
        full_flat_y = self.center_offset + y
        cube_position = self.FaceToCube[cube_id](full_flat_x, full_flat_y)
        return cube_id, np.array(cube_position)

    def convert_to_flat_space(self, p):
        raise NotImplemented()

    def get_flat_from_scale(self, p):
        return np.multiply(p, [self.flat_width*4, self.flat_width])

    def render(self, points):
        import matplotlib.pyplot as plt
        axes = [self.width] * 3
        plt.figure(figsize=(3 * len(points), 3))
        for i, p in enumerate(points):
            cube_id = self.compute_face_index(p[0])
            plt.subplot(1, len(points), i+1)
            # cube
            rectangle = plt.Rectangle((0, 0), self.width, self.width, fc='blue', alpha=0.2)
            plt.gca().add_patch(rectangle)
            # contact range
            rectangle = plt.Rectangle([self.center_offset]*2, self.flat_width, self.flat_width
                                      , fc='orange', alpha=0.5)
            plt.gca().add_patch(rectangle)
            _local_p = np.array(p) - np.array([cube_id*self.flat_width, 0]) + self.center_offset
            plt.scatter(*_local_p, c='r')
            plt.xlim(0, self.width)
            plt.ylim(0, self.width)
            plt.title(f'Face {cube_id}')
            plt.grid()
        plt.tight_layout()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(np.ones(axes), facecolors=[0, 0.5, 0.5, 0.6])
        for p in points:
            cube_p = ccs.convert_to_cube_space(*p)
            world_p = cube_p + np.array(axes) / 2
            ax.scatter(world_p[0], world_p[1], world_p[2],
                       c=[1, 0, 0, 1], linewidths=10)
        plt.tight_layout()
        plt.show()


class ContactPlanner:
    def __init__(self, cube_width=0.065):
        self.contact_map = CubeContactMap(cube_width)   # 0.5

        tip_0_x_bound = [0.25, 0.74]  # tip 0: face 1, 2
        tip_1_x_bound = [0.00, 0.49]  # tip 1: face 0, 1
        tip_2_x_bound = [0.75, 0.99]  # tip 2: face 3

        tips_y_bound = [0.00, 0.99]

        self.action_space = gym.spaces.Box(
            low=np.array([tip_0_x_bound[0],
                          tip_1_x_bound[0],
                          tip_2_x_bound[0],
                          tips_y_bound[0],
                          tips_y_bound[0],
                          tips_y_bound[0]], dtype=np.float32),
            high=np.array([tip_0_x_bound[1],
                           tip_1_x_bound[1],
                           tip_2_x_bound[1],
                           tips_y_bound[1],
                           tips_y_bound[1],
                           tips_y_bound[1]], dtype=np.float32),
        )

    def compute_contact_points(self, action):
        contact_ids = []
        contact_points = []
        for i in range(3):
            flat_point = self.contact_map.get_flat_from_scale([action[i] % 1.0, action[3+i]])   # for 1. > x point
            contact_id, pos = self.contact_map.convert_to_cube_space(flat_point[0], flat_point[1])
            contact_ids.append(contact_id)
            contact_points.append(pos)
        return contact_ids, np.array(contact_points)


if __name__ == '__main__':
    ccs = CubeContactMap(10, center=0.6)
    scale_list = np.array([
        (0.5, 0.99),
        (0, 0.1),
        (0.8, 0.5)
    ])
    point_list = [ccs.get_flat_from_scale(s) for s in scale_list]
    ccs.render(point_list)

