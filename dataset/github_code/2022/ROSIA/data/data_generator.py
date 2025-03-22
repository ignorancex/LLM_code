from simulator import *
import numpy as np
import sys

if __name__ == "__main__":

    ''' ############################### Arguments ############################### '''

    black_list_dir = sys.argv[1]
    out_dir = sys.argv[2]
    dist_noise_sigma = float(sys.argv[3])
    mag_noise_sigma = float(sys.argv[4])
    min_false = float(sys.argv[5])
    mag_thres = float(sys.argv[6])
    num_scenes = int(sys.argv[7])
    hip_cat_filename = sys.argv[8]

    res_x = 1024 # pixels
    res_y = 1024

    # normalized focal length
    FOV = 14
    f = 0.5 / np.tan(np.deg2rad(FOV) / 2)

    # pixel aspect ratio
    pixel_ar = 1

    # normalized principal point
    ppx = 0.5
    ppy = 0.5
    cam = 0

    # magnitude parameters
    A_pixel = 525 # photonelectrons/s mm
    sigma_pixel = 525 # photonelectrons/s mm

    sigma_psf = 0.5 # pixel
    t_exp = 0.2 # s
    aperture = 15 # mm

    base_photons = 19100 # photoelectrons per mm² and second of a magnitude 0 G2 star

    # star count
    min_true = 0
    max_true = 100
    min_stars = 5
    max_false = min_false

    ''' ############################### Arguments ############################### '''

    catalog = StarCatalog(mag_thres, hip_cat_filename)
    hip = catalog.catalog.HIP.values.tolist()

    hip_array = np.zeros([len(hip), 1])
    for i in range(len(hip)):
        hip_array[i] = hip[i]

    cat_star_vec = catalog.star_vectors
    cat_star_mag = catalog.magnitudes
    min_mag = np.min(cat_star_mag)

    cameras = [
        RectilinearCamera,
        EquidistantCamera,
        EquisolidAngleCamera,
        StereographicCamera,
        OrthographicCamera,
    ]

    camera = cameras[cam](f, (res_x, res_y), pixel_ar, (ppx, ppy))

    ''' #### process black_list here ##########'''
    with open(black_list_dir) as f:
        lines = f.read().splitlines()

    blacklist_id = np.zeros([len(lines)]).astype(np.int64)

    for i in range(len(lines)):
        blacklist_id[i] = np.int64(lines[i])

    detector = StarDetector(A_pixel, sigma_pixel, sigma_psf, t_exp, aperture, base_photons)

    for scene_id in range(num_scenes):
        curr_out_dir = out_dir + str(scene_id) + ".txt"
        scene = Scene.random(catalog, camera, detector, min_true, max_true, max_false, max_false, min_stars, mag_thres,
                             gaussian_noise_sigma=dist_noise_sigma, magnitude_gaussian=mag_noise_sigma)

        for i in range(len(scene.pos)):
            if scene.ids[i] not in blacklist_id:
                curr_ang = camera.to_angles(scene.pos[i][np.newaxis, :])
                curr_vec = angles_to_vector(curr_ang[0], curr_ang[1])[0]
                curr_mag = scene.magnitudes[i]
                curr_hip = scene.ids[i]
                tmp_str = str(curr_vec[0]) + " " + str(curr_vec[1]) + " " + str(curr_vec[2]) + " " + str(curr_mag) + " " + str(curr_hip) + "\n"
                f = open(curr_out_dir, "a")
                f.write(tmp_str)

        gt_rotation_matrix = scene.orientation

        GT_rot_file = open(out_dir + "rot_" + str(scene_id) + ".txt", "a")
        GT_rot_file.write(str(gt_rotation_matrix[0, 0]) + " " + str(gt_rotation_matrix[0, 1]) + " " + str(gt_rotation_matrix[0, 2]) +"\n")
        GT_rot_file.write(str(gt_rotation_matrix[1, 0]) + " " + str(gt_rotation_matrix[1, 1]) + " " + str(gt_rotation_matrix[1, 2]) +"\n")
        GT_rot_file.write(str(gt_rotation_matrix[2, 0]) + " " + str(gt_rotation_matrix[2, 1]) + " " + str(gt_rotation_matrix[2, 2]) +"\n")
        GT_rot_file.close()

        f.close()

