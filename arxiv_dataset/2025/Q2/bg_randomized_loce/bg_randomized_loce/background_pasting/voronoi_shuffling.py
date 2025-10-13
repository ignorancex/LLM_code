""""
General setup for voronoi diagram
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.voronoi_plot_2d.html
Generate polygones from voronoi cells
https://gist.github.com/pv/8036995

Draw and fill voronoi diagrams
"""
import os
import random
from typing import Callable

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


# # Ensure polygon points are within bounds
# def clamp_point(point, width, height):
#     x, y = point
#     return max(0, min(x, width - 1)), max(0, min(y, height - 1))


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Subclass of ImageFolder that returns (image, label, image_path) tuple.
    """
    def __getitem__(self, index):
        # Standard behavior: get (image, label)
        original_tuple = super().__getitem__(index)
        # Get the image path
        path = self.imgs[index][0]
        # Return image, label, and path
        return (*original_tuple, path)




def voronoi_finite_polygons_2d(vor, radius=None) -> tuple[list, np.ndarray]:
    """
    Limit infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices),


def randomize_voronoi_diagram(num_points, dim, name, img_to_shuffle, split, rng, result_dir="results", size=(256,256), post_transform: Callable=None, voronoi_offset=100):
    '''
    Generate voronoi centers and fill each cell with a texture image of a certain class

    Parameters
    ----------
    num_points: Number of voronoi centers
    dim: dimension of the diagram (e.g. 2D)
    name: name to identify the generated diagram

    Returns
    ---------
    Voronoi Diagram
    '''
    fig, ax1 = plt.subplots(dpi=300)
    points = rng.random((num_points, dim))*size

    # add 4 distant dummy points
    # points = np.append(points, [[2440, 1440], [-0, 1440], [2440, -0], [-0, -0]], axis=0)
    # points = np.append(points, [[size[0] + 150, size[1] + 150 ], [-0, size[1] + 150], [size[0] + 150, -0], [-0, -0]], axis=0)
    points = np.append(points, [[size[0] + voronoi_offset, size[1] + voronoi_offset ], [-0, size[1] + voronoi_offset], [size[0] + voronoi_offset, -0], [-0, -0]], axis=0)
    vor = Voronoi(points)

    voronoi_plot_2d(vor, ax1, show_vertices=False)
    plt.xlim([0, size[0]]), plt.ylim([0, size[1]])

    # generate Voronoi with finite cells
    regions, vertices = voronoi_finite_polygons_2d(vor)

    voronoi_img = Image.new('RGB', size, 0)
    
    
    for region in regions:
        if region:  # valid region (not empty polygonpoint list) and not -1 in region:
            polygon = vertices[region]

            voronoi_img = cropping(voronoi_img,
                                   img_to_shuffle,
                                   polygon,
                                   size)

    label = os.path.dirname(path).split("/")[-1]
    result_dir = os.path.join(result_dir, split, label)
    os.makedirs(result_dir, exist_ok=True)
    name = os.path.basename(path).removesuffix('.JPEG')
    trasformed_voronoi_img = post_transform(voronoi_img) if post_transform is not None else voronoi_img
    trasformed_voronoi_img.save(os.path.join(result_dir, str(name)) + ".png")
    # plt.gca().invert_yaxis()
    # ax1.plot(points[:-4, 0], points[:-4, 1], 'ro', markersize=2, zorder=10)

    # fig.savefig(f"Images/voronoi_structure_{name}.png")
    plt.close()


def cropping(voronoi: PIL.Image.Image, img_to_paste_from: PIL.Image.Image, polygon: np.ndarray, size: tuple[int, int]) -> PIL.Image.Image:
    '''
    Fill a single Voronoi cell with texture
    '''

    width, height = voronoi.size
    if img_to_paste_from.size != voronoi.size:
        img_to_paste_from = img_to_paste_from.resize(voronoi.size)
    
    # Define the bounding box
    bounding_box = box(0, 0, width, height)
    polygon = Polygon(polygon)

    # Crop the polygon using intersection
    cropped_polygon_points = polygon.intersection(bounding_box)
    if cropped_polygon_points.area == 0.0:
        return voronoi 

    # calculate random shift
    shift = (np.int32(random.uniform(-cropped_polygon_points.bounds[0], size[0] - cropped_polygon_points.bounds[2])),
             np.int32(random.uniform(-cropped_polygon_points.bounds[1], size[1] - cropped_polygon_points.bounds[3])))
    
    shifted_polygon = translate(cropped_polygon_points, xoff=shift[0], yoff=shift[1])

    # polygon_points = np.array([clamp_point(p, width, height) for p in polygon])


    polygon_filled = Image.new('RGB', size, 0)
    polygon_mask_shifted = Image.new('L', size, 0)

    polygon_mask = Image.new('L', size, 0)
    polygon_path_shifted = list(shifted_polygon.exterior.coords)
    polygon_path =list(cropped_polygon_points.exterior.coords)
    ImageDraw.Draw(polygon_mask_shifted).polygon(polygon_path_shifted, fill=(255), outline=(255))
    ImageDraw.Draw(polygon_mask).polygon(polygon_path, fill=(255), outline=(255))

    #fill voronoi diagram
    polygon_filled.paste(img_to_paste_from, (0, 0), polygon_mask_shifted)
    voronoi.paste(polygon_filled, (-shift[0], -shift[1]), polygon_mask_shifted)

    return voronoi



# TEST SETUP
if __name__ == "__main__":
    # INFO: scripts expect a certain naming convention as implemented in polygon_semseg_contourfill.py
    # Adaption of the function "pathfind" is needed if contour filling is skipped in the procedure 
    cell_number = 32
    dimensionen = 2
    diagramm_amount = 10350
    size = (256, 256)
    root_path = "/home/eheinert/data/EED_benchmark_datasets/ImageNet_subsample_asGeirhos_16_classes"
    base_prefix = "imagenet"
    split = 'val'
    result_dir = f"Voronoi_{base_prefix}"
    # label_list = [0,1,2,5,6,7,8,10,11,13,18]
    label_list = []

    rng = np.random.default_rng(4224)  # random number generator for random handling in numpy

    dataset_to_shuffle = ImageFolderWithPaths(root=root_path, transform=transforms.Resize(256))
    post_transformation = transforms.CenterCrop(224)

    for img_to_shuffle, label_id, path in tqdm(dataset_to_shuffle):
        randomize_voronoi_diagram(cell_number,
                                  dimensionen,
                                  name=path,
                                  split=split,
                                  rng = rng,
                                  img_to_shuffle=img_to_shuffle,
                                  result_dir=result_dir,
                                  size=img_to_shuffle.size,
                                  post_transform=post_transformation
                                  )
# end main