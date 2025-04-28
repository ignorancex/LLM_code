import numpy as np
import os
import SimpleITK as sitk
import random
import json
import argparse
import sys


class DataSyn:
    def __init__(
            self,
            source_dir,
            target_dir,
    ):
        """
        source_dir: The directory of the raw data.
        target_dir: The directory of the data to be synthesized.

        Input: source_dir/
                - pulse_xxxxx_volume.nii.gz  % original volume
                - pulse_xxxxx_graph.json  % graph with edge information
                ...

        Output: target_dir/
                - pulse_xxxxx_idx_volume.nii.gz  % synthesized volume with single disconnection
                - pulse_xxxxx_idx_kp1_part.nii.gz  % kp1 part of the synthesized volume
                - pulse_xxxxx_idx_kp2_part.nii.gz  % kp2 part of the synthesized volume
                - pulse_xxxxx_idx_data.npz  % meta data of the synthesized volume
                ...

        Note: centerline infomation will not be used in this process.
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.pulse_ids = [i[:11] for i in os.listdir(source_dir) if '.json' in i]

    def data_process(self, pulse_id, volume, edge, idx):
        point_coord = np.array(edge['point_coord'])
        point_number = edge['point_number']
        tolerance = 0
        flag = 0
        while True:
            # choose the location of KP1 and KP2
            kp1_range = [0.05, 0.4]
            kp2_range = [0.6, 0.95]
            kp1_location = np.random.random() * (kp1_range[1] - kp1_range[0]) + kp1_range[0]
            kp2_location = np.random.random() * (kp2_range[1] - kp2_range[0]) + kp2_range[0]
            kp1 = point_coord[round(kp1_location * point_number)]
            kp2 = point_coord[round(kp2_location * point_number)]
            kp_mid = point_coord[round(0.5 * point_number)]
            coords = np.argwhere(volume != 0)

            edge_points_dif = point_coord[1:] - point_coord[:-1]  # point_x - point_(x+1)
            edge_points_dis = np.sqrt(np.sum(edge_points_dif ** 2, axis=1))
            edge_points_toend_dis = []
            for i in range(point_number):
                if i == 0:
                    edge_points_toend_dis.append(0.0)
                elif i == point_number - 1:
                    edge_points_toend_dis.append(0.0)
                else:
                    edge_points_toend_dis.append(min(np.sum(edge_points_dis[:i]), np.sum(edge_points_dis[i:])))
            edge_points_toend_dis = np.array(edge_points_toend_dis)

            vector_1 = kp_mid - kp1
            vector_2 = kp_mid - kp2
            radius_cutoff = np.random.random() * (2 - 1.3) + 1.3
            noise_level = np.random.random() * (4 - 1) + 1

            for p in coords:
                if (p - kp1).dot(vector_1) > 0 and (p - kp2).dot(vector_2) > 0:
                    distance = np.sum((p - point_coord) ** 2, axis=1)
                    closest_index = np.argmin(distance)

                    if distance[closest_index] < (edge['radius_max'] * radius_cutoff) ** 2:
                        volume[p[0], p[1], p[2]] = 0
                        ratio = 1 - (edge_points_toend_dis[closest_index] / np.max(edge_points_toend_dis))

                        if random.random() < ratio ** noise_level * 0.5:
                            volume[p[0], p[1], p[2]] = 1

            volume_img = sitk.GetImageFromArray(volume)
            component_image = sitk.ConnectedComponent(volume_img)
            sorted_component_image = sitk.RelabelComponent(component_image, minimumObjectSize=15, sortByObjectSize=True)
            sorted_component = sitk.GetArrayFromImage(sorted_component_image)

            # label_0: the label of the kp1 component; label_1: the label of the kp2 component
            label_0 = sorted_component[kp1[0], kp1[1], kp1[2]]
            label_1 = sorted_component[kp2[0], kp2[1], kp2[2]]

            # label=0: background
            if label_0 * label_1 > 0 and label_0 != label_1:
                seg_0 = sorted_component == label_0
                seg_1 = sorted_component == label_1
                disconnected_volume = np.logical_or(seg_0, seg_1) + 0
                seg_0_size = np.sum(seg_0)
                seg_1_size = np.sum(seg_1)

                # make sure kp1 locate at the main part of the pulmonary tree
                if seg_0_size > seg_1_size:
                    kp1_seg = seg_0 + 0
                    kp2_seg = seg_1 + 0
                    kp1_coord = kp1
                    kp2_coord = kp2
                else:
                    kp1_seg = seg_1 + 0
                    kp2_seg = seg_0 + 0
                    kp1_coord = kp2
                    kp2_coord = kp1

                disconnected_volume_nii = sitk.GetImageFromArray(np.uint8(disconnected_volume))
                sitk.WriteImage(disconnected_volume_nii, os.path.join(self.target_dir, pulse_id) +
                                '_'+str(idx)+'_volume.nii.gz')

                kp1_volume_nii = sitk.GetImageFromArray(np.uint8(kp1_seg))
                sitk.WriteImage(kp1_volume_nii, os.path.join(self.target_dir, pulse_id) +
                                '_' + str(idx) + '_kp1_part.nii.gz')

                kp2_volume_nii = sitk.GetImageFromArray(np.uint8(kp2_seg))
                sitk.WriteImage(kp2_volume_nii, os.path.join(self.target_dir, pulse_id) +
                                '_' + str(idx) + '_kp2_part.nii.gz')

                npz_name = os.path.join(self.target_dir, pulse_id + '_' + str(idx) + '_data.npz')
                if not os.path.exists(npz_name):
                    np.savez_compressed(npz_name,
                                        volume=disconnected_volume,
                                        kp1_part=kp1_seg,
                                        kp2_part=kp2_seg,
                                        kp1_coord=kp1_coord,
                                        kp2_coord=kp2_coord,
                                        edge_points=point_coord,  # (n,3)
                                        edge_radius_avg=edge['radius_avg'],
                                        edge_radius_min=edge['radius_min'],
                                        edge_radius_max=edge['radius_max'],
                                        edge_volume=edge['volume'],
                                        edge_surface_area=edge['surface_area'],
                                        edge_length=edge['length'],
                                        edge_tortuosity=edge['tortuosity']
                                        )
                print('{} has been processed!'.format(pulse_id + '_' + str(idx)))
                flag = 1
                break
            else:
                if tolerance > 10:
                    print('{} does not work!!!'.format(pulse_id))
                    break
                tolerance += 1
        return flag

    def data_synthesis(self, volume_num, radius_range, points_threshold):
        for pulse_id in self.pulse_ids:
            # read json
            json_file = os.path.join(self.source_dir, pulse_id+'_graph.json')
            with open(json_file, 'r') as f:
                graph = json.load(f)

            # read volume.nii.gz
            volume_file = os.path.join(self.source_dir,pulse_id+'_volume.nii.gz')
            volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_file))  # uint8

            # random selection with a condition
            graph_edges = list(graph)
            random.shuffle(graph_edges)
            idx = 0
            while True:
                edge = graph[graph_edges.pop(0)]
                if edge['point_number'] >= points_threshold:
                    if radius_range[0] <= edge['radius_avg'] <= radius_range[1]:
                        flag = self.data_process(pulse_id, volume, edge, idx)
                        if flag:
                            idx += 1
                if idx >= volume_num:
                    break


def main_parser(args=sys.argv[1:]):
    # Parser definition
    parser = argparse.ArgumentParser(description="Parses command.")

    # Parser Options
    parser.add_argument("-source_dir", default='raw_data/', help="Path of the raw data")
    parser.add_argument("-target_dir", default='synthesized_data/', help="Path of the synthesized data")
    parser.add_argument("-volume_num", type=int, help="Set number of synthesized disconnected volumes per subject")
    parser.add_argument("-radius_min", type=float, default=0.,
                        help="Set the minimum radius of the edge to be disconnected")
    parser.add_argument("-radius_max", type=float, default=50.,
                        help="Set the maximum radius of the edge to be disconnected")
    parser.add_argument("-points_threshold", type=int, default=10.,
                        help="Edges with fewer points than this threshold will be filtered")
    options = parser.parse_args(args)

    if options.radius_min >= options.radius_max:
        parser.error("--radius_min cannot be equal or lager than radius_max")

    return options


if __name__ == '__main__':
    """
    A simple command to run this script:
     
    python Data_Synthesis.py -source_dir raw_data/ -target_dir synthesized_data/ -volume_num=2 -radius_min=1 -radius_max=15 -points_threshold=10
    """
    options = main_parser(sys.argv[1:])
    if len(sys.argv) == 1:
        print("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()
        
    if not os.path.exists(options.target_dir):
        os.mkdir(options.target_dir)

    generater = DataSyn(source_dir=options.source_dir,
                        target_dir=options.target_dir)
    generater.data_synthesis(volume_num=options.volume_num,
                             radius_range=[options.radius_min, options.radius_max],
                             points_threshold=options.points_threshold)

