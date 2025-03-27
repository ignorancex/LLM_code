import bpy

import sys, os

from pathlib import Path

from mathutils import Matrix, Vector

import math

import numpy as np

import json

from numpy import linalg as la
    
################### FOR OUR METHOD ################################

scene_name = 'old_union'
method = 'nerfnav'
base = Path(bpy.path.abspath('//'))
parent = base.parent

data_filepath = os.path.join(parent, f'trajs/{scene_name}_{method}.json')

my_coll = bpy.data.collections.new(f'{scene_name}')
bpy.context.scene.collection.children.link(my_coll)
# material = bpy.data.materials["Material.004"]

#start_obj = bpy.data.objects['start']
#end_obj = bpy.data.objects['end']

with open(data_filepath, 'r') as f:
    meta = json.load(f)

datum = meta["total_data"]
robot_radius = meta["radius"]

for i, data in enumerate(datum):
    traj = data["traj"]
    if len(traj) > 2:
        traj = np.array(traj)[..., :3]

#        #FOR AXES
#        start_pt = bpy.data.objects.new(f'start_point{iter}', start_obj.data)
#        start_pt.scale = [0.04, 0.04, 0.04]
#        my_coll.objects.link(start_pt)

#        end_pt = bpy.data.objects.new(f'end_point{iter}', end_obj.data)
#        end_pt.scale = [0.04, 0.04, 0.04]
#        my_coll.objects.link(end_pt)

#        start = traj[0, :3]
#        end = traj[-1, :3]

#        start_pt.location = [start[0], start[1], start[2]]
#        end_pt.location = [end[0], end[1], end[2]]
        bpy.context.view_layer.update()

        # make a new curve
        crv = bpy.data.curves.new('crv', 'CURVE')
        crv.dimensions = '3D'

        # make a new spline in that curve
        spline = crv.splines.new(type='POLY')

        # a spline point for each point
        spline.points.add(len(traj)-1) # theres already one point by default

        # assign the point coordinates to the spline points
        for p, new_co in zip(spline.points, traj[:, :3]):
            p.co = (new_co.tolist() + [1.0]) # (add nurbs weight)

        # make a new object with the curve
        obj = bpy.data.objects.new(f'traj_{i}', crv)
        obj.data.bevel_depth = robot_radius
        # obj.data.materials.append(material)
        my_coll.objects.link(obj)


print("--------------------    DONE WITH BLENDER SCRIPT    --------------------")