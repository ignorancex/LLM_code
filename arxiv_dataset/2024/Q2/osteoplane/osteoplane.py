import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import math
# from mpl_toolkits.mplot3d import Axes3D
# from quaternion import as_euler_angles
# from scipy.spatial.transform import Rotation as R
# from plyfile import PlyData, PlyElement
import tkinter as tk     # from tkinter import Tk for Python 3.x
import tkinter.filedialog as fd # askopenfilename
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches
import os
import plotly.graph_objects as go
import argparse
import plot_utils
import model_utils

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")


parser.add_argument('-s', '--segments', type=int, help='Number of segments')
parser.add_argument('-i', '--id', type=str, help='ID')

# Parse the arguments
args = parser.parse_args()

# You can then access the arguments as follows:
segments_arg = args.segments
id = args.id


"visualize np array of points as point cloud"
def open3DPick(file, density):
    mesh = o3d.io.read_triangle_mesh(file, density)
    "convert mesh to point cloud and optionally visualize"
    #visualize Mesh
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([.5, .5, .5])
    # o3d.visualization.draw_geometries([mesh])
    pcd = mesh.sample_points_uniformly(number_of_points=density)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    index = vis.get_picked_points()    
    pcd_array = np.asarray(pcd.points)
    return index, pcd_array, pcd

"visualize np array of points as point cloud"
def open3DNoPick(file, density):
    mesh = o3d.io.read_triangle_mesh(file, density)
    "convert mesh to point cloud and optionally visualize"
    #visualize Mesh
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([.5, .5, .5])
    # o3d.visualization.draw_geometries([mesh])
    pcd = mesh.sample_points_uniformly(number_of_points=density)
    return pcd

"initial file selection"
def fileSelect():
    root = tk.Tk()
    files = fd.askopenfilenames(parent=root, title='Choose a file')
    file_list = list(files)
    print(file_list)
    return (file_list)

"do all the open 3d selection and arrangement of data to be used in main loop"
def data_gathering(preop, postop, density, segments):
    preop_prox_seg_index, preop_arr, preop_cloud = open3DPick(preop, density)
    target_1_A = preop_arr[preop_prox_seg_index[0]]
    target_1_B = preop_arr[preop_prox_seg_index[1]]
    target_1_C = preop_arr[preop_prox_seg_index[2]] 
    target_1_plane = [target_1_A.tolist(),target_1_B.tolist(),target_1_C.tolist()] 
    
    target_2_A = preop_arr[preop_prox_seg_index[3]]
    target_2_B = preop_arr[preop_prox_seg_index[4]]
    target_2_C = preop_arr[preop_prox_seg_index[5]]
    target_2_plane = [target_2_A.tolist(),target_2_B.tolist(),target_2_C.tolist()] 
    
    plan_distance_seg_1 = np.linalg.norm(target_1_A - target_2_A)
    
    target_3_A = preop_arr[preop_prox_seg_index[6]]
    target_3_B = preop_arr[preop_prox_seg_index[7]]
    target_3_C = preop_arr[preop_prox_seg_index[8]] 
    target_3_plane = [target_3_A.tolist(),target_3_B.tolist(),target_3_C.tolist()] 
    
    plan_distance_seg_2 = np.linalg.norm(target_2_A - target_3_A)

    if segments == 3:
        target_4_A = preop_arr[preop_prox_seg_index[9]]
        target_4_B = preop_arr[preop_prox_seg_index[10]]
        target_4_C = preop_arr[preop_prox_seg_index[11]]
        target_4_plane = [target_4_A.tolist(),target_4_B.tolist(),target_4_C.tolist()] 
        plan_distance_seg_3 = np.linalg.norm(target_3_A - target_4_A)


    postop_prox_seg_index, postop_arr, postop_cloud = open3DPick(postop, density)
    postOpProxA_1 = postop_arr[postop_prox_seg_index[0]]
    postOpProxB_1 = postop_arr[postop_prox_seg_index[1]]
    postOpProxC_1 = postop_arr[postop_prox_seg_index[2]]
    plane_post_prox_1 = [postOpProxA_1.tolist(),postOpProxB_1.tolist(),postOpProxC_1.tolist()] 
    
    postOpDistA_1 = postop_arr[postop_prox_seg_index[3]] 
    postOpDistB_1 = postop_arr[postop_prox_seg_index[4]]   
    postOpDistC_1 = postop_arr[postop_prox_seg_index[5]] 
    plane_post_dist_1 = [postOpDistA_1.tolist(),postOpDistB_1.tolist(),postOpDistC_1.tolist()] 

    post_distance_seg_1 = np.linalg.norm(postOpProxA_1 - postOpDistA_1)

    postOpProxA_2 = postop_arr[postop_prox_seg_index[6]]
    postOpProxB_2 = postop_arr[postop_prox_seg_index[7]]
    postOpProxC_2 = postop_arr[postop_prox_seg_index[8]]
    plane_post_prox_2 = [postOpProxA_2.tolist(),postOpProxB_2.tolist(),postOpProxC_2.tolist()] 
    
    postOpDistA_2 = postop_arr[postop_prox_seg_index[9]] 
    postOpDistB_2 = postop_arr[postop_prox_seg_index[10]]   
    postOpDistC_2 = postop_arr[postop_prox_seg_index[11]] 
    plane_post_dist_2 = [postOpDistA_2.tolist(),postOpDistB_2.tolist(),postOpDistC_2.tolist()] 
    
    post_distance_seg_2 = np.linalg.norm(postOpProxA_2 - postOpDistA_2)
    
    if segments == 3:
        postOpProxA_3 = postop_arr[postop_prox_seg_index[12]]
        postOpProxB_3 = postop_arr[postop_prox_seg_index[13]]
        postOpProxC_3 = postop_arr[postop_prox_seg_index[14]]
        plane_post_prox_3 = [postOpProxA_3.tolist(),postOpProxB_3.tolist(),postOpProxC_3.tolist()] 
        
        postOpDistA_3 = postop_arr[postop_prox_seg_index[15]] 
        postOpDistB_3 = postop_arr[postop_prox_seg_index[16]]   
        postOpDistC_3 = postop_arr[postop_prox_seg_index[17]] 
        plane_post_dist_3 = [postOpDistA_3.tolist(),postOpDistB_3.tolist(),postOpDistC_3.tolist()] 
        post_distance_seg_3 = np.linalg.norm(postOpProxA_3 - postOpDistA_3)

    
    segment_1_plan = [target_1_plane,target_2_plane]
    segment_1_post = [plane_post_prox_1,plane_post_dist_1]
    
    segment_1 = [segment_1_plan, segment_1_post]
    
    segment_2_plan = [target_2_plane,target_3_plane]
    segment_2_post = [plane_post_prox_2,plane_post_dist_2]
    
    segment_2 = [segment_2_plan, segment_2_post]

    final_segments = [segment_1, segment_2 ]
    # final_segments = [[[xyz], [xyz]], [[xyz],[xyz]]   ,  [] []      ]
    if segments==3:
        segment_3_plan = [target_3_plane, target_4_plane]
        segment_3_post = [plane_post_prox_3,plane_post_dist_3]
        segment_3 = [segment_3_plan, segment_3_post]
        final_segments = [segment_1, segment_2, segment_3]


    distances.append(plan_distance_seg_1)
    distances.append(plan_distance_seg_2)
    if segments ==3:
        distances.append(plan_distance_seg_2)
    distances.append(post_distance_seg_1)
    distances.append(post_distance_seg_2)
    if segments ==3:
        distances.append(post_distance_seg_3)
    return final_segments, preop_cloud, postop_cloud

def main_loop(preop, postop, density, segments):
    final_segments, preop_cloud, postop_cloud = data_gathering(preop, postop, density, segments)
    for idx, segment in enumerate(final_segments):
        ### PLOT HERE segment[0] and segment[1]
        plan_model = segment[0]
        postop_model = segment[1]
         
        plan_model_prox = plan_model[0]
        postop_model_prox = postop_model[0]

        # visualize_data(plan_model, postop_model, idx+1)
        translated_segment_plan = model_utils.moveToOrigin(plan_model, plan_model_prox[0])
        angle1_preop = model_utils.getAngle_X(translated_segment_plan[1][0])
        
        translated_segment_postop = model_utils.moveToOrigin(postop_model, postop_model_prox[0])
        angle1_postop = model_utils.getAngle_X(translated_segment_postop[1][0])
    
        # visualize_data(translated_segment_plan, translated_segment_postop, idx+2)
        
        rotate_1_preop = model_utils.rotate_struct_1(translated_segment_plan, angle1_preop)
        rotate_1_postop = model_utils.rotate_struct_1(translated_segment_postop, angle1_postop)
        
        # visualize_data(rotate_1_preop, rotate_1_postop,idx+3)
        

        
        angle2_preop = model_utils.getAngle_Z(rotate_1_preop[1][0])
        angle2_postop = model_utils.getAngle_Z(rotate_1_postop[1][0])

        rotate_2_preop = model_utils.rotate_struct_2(rotate_1_preop, angle2_preop)
        rotate_2_postop = model_utils.rotate_struct_2(rotate_1_postop, angle2_postop)
        
        plot_utils.visualize_data(rotate_2_preop, rotate_2_postop, idx+1)
        

        #define plane equatinos for proximal and distal pre and post op planes
        preProxA1,preProxB1,preProxC1 = model_utils.definePlaneEquation(rotate_2_preop[0],'one')
        preDistA1,preDistB1,preDistC1 = model_utils.definePlaneEquation(rotate_2_preop[1],'one')

        postProxA1,postProxB1,postProxC1 = model_utils.definePlaneEquation(rotate_2_postop[0],'one')
        postDistA1,postDistB1,postDistC1 = model_utils.definePlaneEquation(rotate_2_postop[1],'one')

        # normal plane should be xz plane- used three random points from xz plane. Need d value 
        # (distance from xz plane.) In this case, d = 1/2 the y component of the segment not on the origin


        
        y_pre = rotate_2_preop[1][0][1] /2
        y_post = rotate_2_postop[1][0][1] /2
        
        # subtract normal plane - plane 1 and normal plane - plane 2 to get delta (second metric of reporting)
        normMinusPlaneProxPre = model_utils.calcPlaneAngle(preProxA1,preProxB1,preProxC1,0,y_pre,0)
        normMinusPlaneDistPre = model_utils.calcPlaneAngle(preDistA1,preDistB1,preDistC1,0,y_pre,0)

        normMinusPlaneProxPost = model_utils.calcPlaneAngle(postProxA1,postProxB1,postProxC1,0,y_post,0)
        normMinusPlaneDistPost = model_utils.calcPlaneAngle(postDistA1,postDistB1,postDistC1,0,y_post,0)

        deltaProx = normMinusPlaneProxPost-normMinusPlaneProxPre
        deltaDist = normMinusPlaneDistPost-normMinusPlaneDistPre

        deltas.append(abs(deltaProx))
        deltas.append(abs(deltaDist))
        
        v1 = np.array([preProxA1,preProxB1,preProxC1])
        v2 = np.array([postProxA1,postProxB1,postProxC1])
        
        phi1, theta1, psi1, r1 = model_utils.get_euler_angles(v2, v1)
        v3 = np.dot(r1, v2)
        a_prox, b_prox, c_prox = v3
        plot_utils.plot_planes(v1, v2, v3, idx+1, 'Proximal')
        
        v4 = np.array([preDistA1,preDistB1,preDistC1])
        v5 = np.array([postDistA1,postDistB1,postDistC1])
        phi2,theta2, psi2, r2 = model_utils.get_euler_angles(v5, v4)
        v6 = np.dot(r2, v5)
        a_dist, b_dist, c_dist = v6
        plot_utils.plot_planes(v4, v5, v6, idx+1, 'Distal')

        proximal_angles = [abs(phi1), abs(theta1), abs(psi1)]  
        distal_angles = [abs(phi2), abs(theta2), abs(psi2)]  

        plot_utils.plot_radar_chart( [abs(phi1), abs(theta1), abs(psi1)] ,  [abs(phi2), abs(theta2), abs(psi2)]  , idx+1)
        plot_utils.euler_bar_chart(proximal_angles, distal_angles, idx+1)
        
        
        pre_prox_equation = model_utils.roundPlanes(preProxA1, preProxB1, preProxC1)
        pre_dist_equation = model_utils.roundPlanes(preDistA1, preDistB1, preDistC1)
        
        post_prox_equation = model_utils.roundPlanes(postProxA1, postProxB1, postProxC1)
        post_dist_equation = model_utils.roundPlanes(postDistA1, postDistB1, postDistC1)
        
        proximal_new_equation = model_utils.roundPlanes(a_prox, b_prox, c_prox)
        distal_new_equation = model_utils.roundPlanes(a_dist, b_dist, c_dist)
        
        pre_mesh_prox = model_utils.create_triangle_mesh(segment[0][0], color = [1,0,1])
        pre_mesh_dist = model_utils.create_triangle_mesh(segment[0][1], color = [0,1,1])

        post_mesh_prox = model_utils.create_triangle_mesh(segment[1][0], color = [1,0,1])
        post_mesh_dist = model_utils.create_triangle_mesh(segment[1][1],color = [0,1,1])

        ply_Planes.append(pre_mesh_prox)
        ply_Planes.append(pre_mesh_dist)
        ply_Planes.append(post_mesh_prox)
        ply_Planes.append(post_mesh_dist)
        
        angles.append(abs(round(np.degrees(phi1), 2)))
        angles.append(abs(round(np.degrees(theta1),2 )))
        angles.append(abs(round(np.degrees(psi1),2)))
        
        angles.append(abs(round(np.degrees(phi2),2)))
        angles.append(abs(round(np.degrees(theta2),2)))
        angles.append(abs(round(np.degrees(psi2),2)))
        
        original_planes.append(pre_prox_equation)
        original_planes.append(pre_dist_equation)
        original_planes.append(post_prox_equation)
        original_planes.append(post_dist_equation)
        rotated_planes.append(proximal_new_equation)
        rotated_planes.append(distal_new_equation)
        

    return preop_cloud, postop_cloud
   
density = 1000000

file_list = fileSelect()

preop = file_list[0]
postop = file_list[1]

# segments = input('select number of segments')
segments = segments_arg
initials = id
print(f'Number of segments is  {segments}')

out = '/Users/georgienahass/Desktop/alkureishiLab/osteoplane_proj/osteoplane/figures/'
filename = initials+'osteoplane.csv'

original_planes = []
rotated_planes = []
angles = []
ply_Planes = []
distances = []
deltas = []
if segments == 3:
    with open(out + filename,'w') as fout:
        line = ['segment 1 proximal preop plane','segment 1 distal preop plane','segment 1 proximal postop plane','segment 1 distal postop plane',
            'segment 2 proximal preop plane','segment 2 distal preop plane','segment 2 proximal postop plane','segment 2 distal postop plane',
            'segment 3 proximal preop plane','segment 3 distal preop plane','segment 3 proximal postop plane','segment 3 distal postop plane',
            'NEW segment 1 proximal postop plane','NEW segment 1 distal postop plane',
            'NEW segment 2 proximal postop plane','NEW segment 2 distal postop plane',   
            'NEW segment 3 proximal postop plane','NEW segment 3 distal postop plane',  
            'phi proximal segment 1','theta proximal segment 1','psi proximal segment 1',
            'phi distal segment 1','theta distal segment 1','psi distal segment 1',
            'phi proximal segment 2','theta proximal segment 2','psi proximal segment 2',
            'phi distal segment 2','theta distal segment 2','psi distal segment 2',
            'phi proximal segment 3','theta proximal segment 3','psi proximal segment 3',
            'phi distal segment 3','theta distal segment 3','psi distal segment 3',
            'distance segment 1 preop', 'distance segment 2 preop', 'distance segment 3 preop',
            'distance segment 1 postop', 'distance segment 2 postop', 'distance segment 3 postop',  
            'delta segment 1 proximal', 'delta segment 1 distal',
            'delta segment 2 proximal', 'delta segment 2 distal',
            'delta segment 3 proximal', 'delta segment 3 distal']
        fout.write("%s\n" % ",".join(line))
    
    _, _ = main_loop(preop, postop, density, segments)
    
    low_res_preop = open3DNoPick(preop, 100000)
    low_res_posteop = open3DNoPick(postop, 100000)
    plot_utils.plot_deltas(deltas, segments)
    preop_save = low_res_preop+ply_Planes[0]+ply_Planes[1]+ply_Planes[4]+ ply_Planes[5] + ply_Planes[8] + ply_Planes[9]
    postop_save = low_res_posteop + ply_Planes[2] + ply_Planes[3] + ply_Planes[6] + ply_Planes[7] + ply_Planes[10] + ply_Planes[11]
    
    o3d.io.write_point_cloud('figures/result_plan.ply',preop_save)
    o3d.io.write_point_cloud('figures/result_post-op.ply',postop_save)

    # o3d.visualization.draw_geometries([low_res_preop, ply_Planes[0], ply_Planes[1] ,ply_Planes[4] , ply_Planes[5] ,ply_Planes[8] , ply_Planes[9]])
    # o3d.visualization.draw_geometries([low_res_posteop, ply_Planes[2], ply_Planes[3] ,ply_Planes[6] , ply_Planes[7] ,ply_Planes[10] , ply_Planes[11]])

    with open(out + filename,'a') as fout:
        line = [original_planes[0], original_planes[1], original_planes[2], original_planes[3],  
            original_planes[4], original_planes[5], original_planes[6], original_planes[7], 
            original_planes[8], original_planes[9], original_planes[10], original_planes[11], 
            rotated_planes[0],  rotated_planes[1],  
            rotated_planes[2],  rotated_planes[3], 
            rotated_planes[4],  rotated_planes[5], 
            angles[0], angles[1],  angles[2],
            angles[3], angles[4],  angles[5],
            angles[6], angles[7],  angles[8],
            angles[9], angles[10],  angles[11],
            angles[12], angles[13],  angles[14],
            angles[15], angles[16],  angles[17],
            distances[0], distances[2], distances[4],
            distances[1], distances[3], distances[5],
            deltas[0], deltas[1],
            deltas[2], deltas[3],
            deltas[4], deltas[5]]
        line = [str(l) for l in line]
        fout.write("%s\n" % ",".join(line))
        
if segments == 2:
    with open(out + filename,'w') as fout:
        line = ['segment 1 proximal preop plane','segment 1 distal preop plane','segment 1 proximal postop plane','segment 1 distal postop plane',
            'segment 2 proximal preop plane','segment 2 distal preop plane','segment 2 proximal postop plane','segment 2 distal postop plane',
            'NEW segment 1 proximal postop plane','NEW segment 1 distal postop plane',
            'NEW segment 2 proximal postop plane','NEW segment 2 distal postop plane',   
            'phi proximal segment 1','theta proximal segment 1','psi proximal segment 1',
            'phi distal segment 1','theta distal segment 1','psi distal segment 1',
            'phi proximal segment 2','theta proximal segment 2','psi proximal segment 2',
            'phi distal segment 2','theta distal segment 2','psi distal segment 2',
            'distance segment 1 preop', 'distance segment 2 preop', 
            'distance segment 1 postop', 'distance segment 2 postop',
            'delta segment 1 proximal', 'delta segment 1 distal',
            'delta segment 2 proximal', 'delta segment 2 distal']
        fout.write("%s\n" % ",".join(line))
        
    _, _ = main_loop(preop, postop, density, segments)
    
    low_res_preop = open3DNoPick(preop, 100000)
    low_res_posteop = open3DNoPick(postop, 100000)
    plot_utils.plot_deltas(deltas, segments)

    preop_save = low_res_preop+ply_Planes[0]+ply_Planes[1]+ply_Planes[4]+ ply_Planes[5]
    postop_save = low_res_posteop + ply_Planes[2] + ply_Planes[3] + ply_Planes[6] + ply_Planes[7]
    
    o3d.io.write_point_cloud('figures/result_plan.ply',preop_save)
    o3d.io.write_point_cloud('figures/result_post-op.ply',postop_save)

    # o3d.visualization.draw_geometries([low_res_preop, ply_Planes[0], ply_Planes[1], ply_Planes[4], ply_Planes[5]])
    # o3d.visualization.draw_geometries([low_res_posteop, ply_Planes[2], ply_Planes[3], ply_Planes[6], ply_Planes[7]])

    with open(out + filename,'a') as fout:
            line = [original_planes[0], original_planes[1], original_planes[2], original_planes[3],  
                original_planes[4], original_planes[5], original_planes[6], original_planes[7], 
                rotated_planes[0],  rotated_planes[1],  
                rotated_planes[2],  rotated_planes[3], 
                angles[0], angles[1],  angles[2],
                angles[3], angles[4],  angles[5],
                angles[6], angles[7],  angles[8],
                angles[9], angles[10],  angles[11],
                distances[0], distances[2],
                distances[1], distances[3],
                deltas[0], deltas[1],
                deltas[2], deltas[3]]
            line = [str(l) for l in line]
            fout.write("%s\n" % ",".join(line))




