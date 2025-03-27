import bpy
import sys
from mathutils import Matrix
import json
import numpy as np

# Where to look and where to save
arg_path = bpy.path.abspath('//') + 'renders/transforms.json'
save_path = bpy.path.abspath('//') + 'renders/mesh'

scene = bpy.context.scene
camera = bpy.data.objects['Camera']

try:
    with open(arg_path,"r") as f:
        meta = json.load(f)
except Exception as err:
    print(f"Unexpected {err}, {type(err)}")
    raise

poses = np.array(meta['transform_matrix'])

for i, pose in enumerate(poses[:2]):
    camera.matrix_world = Matrix(pose)
    bpy.context.view_layer.update()

    # save image from camera
    bpy.context.scene.render.filepath = save_path + f'/render_{i}.png'
    bpy.ops.render.render(write_still = True)