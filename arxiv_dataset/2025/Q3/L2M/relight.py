import bpy
import numpy as np
import cv2
import math
import os
import imutils
import scipy.ndimage
from depth_anything_v2.dpt import DepthAnythingV2

# conda activate pgdvs


def resize_and_center_crop(image, disparity):
    # 获取图像和视差图的尺寸
    h, w = image.shape[:2]

    # 计算最短边的尺寸
    shortest_edge = min(h, w)

    # 按最短边缩放
    if h < w:
        new_h = shortest_edge
        new_w = int(shortest_edge * (w / h))
    else:
        new_w = shortest_edge
        new_h = int(shortest_edge * (h / w))

    # 缩放图像
    image_resized = cv2.resize(image, (new_w, new_h))
    disparity_resized = cv2.resize(disparity, (new_w, new_h))

    # 计算裁剪区域，使得图像变为正方形
    crop_size = min(image_resized.shape[:2])  # 取缩放后图像的最短边作为裁剪大小
    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2

    # 裁剪图像和视差图
    image_cropped = image_resized[
        start_y : start_y + crop_size, start_x : start_x + crop_size
    ]
    disparity_cropped = disparity_resized[
        start_y : start_y + crop_size, start_x : start_x + crop_size
    ]

    return image_cropped, disparity_cropped


def create_mesh(vertices, faces, colors, mesh_name="GeneratedMesh"):
    # 创建网格和对象
    mesh_data = bpy.data.meshes.new(mesh_name)
    mesh_object = bpy.data.objects.new(mesh_name, mesh_data)
    bpy.context.collection.objects.link(mesh_object)

    # 设置顶点和面
    mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh_data.update()

    # 添加顶点颜色数据
    if not mesh_data.vertex_colors:
        mesh_data.vertex_colors.new(name="Col")
    color_layer = mesh_data.vertex_colors["Col"]

    # 设置每个顶点的颜色
    for poly in mesh_data.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh_data.loops[loop_index].vertex_index
            color_layer.data[loop_index].color = (*colors[vertex_index], 1.0)  # RGBA

    # 添加材质并启用漫反射着色器
    if not mesh_object.data.materials:
        material = bpy.data.materials.new("MeshMaterial")
        material.use_nodes = True  # 使用节点着色
        mesh_object.data.materials.append(material)

    # 获取材质的节点树
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # 删除默认的 Principled BSDF 节点
    for node in nodes:
        if node.type == "BSDF_PRINCIPLED":
            nodes.remove(node)

    # 添加 Diffuse BSDF 节点
    diffuse_bsdf = nodes.new(type="ShaderNodeBsdfDiffuse")

    # 创建顶点颜色节点
    vertex_color_node = nodes.new("ShaderNodeAttribute")
    vertex_color_node.attribute_name = "Col"  # 顶点颜色的名称
    links.new(vertex_color_node.outputs[0], diffuse_bsdf.inputs["Color"])

    # 将 Diffuse BSDF 节点连接到材质的 Surface 输入
    material_output = nodes.get("Material Output")
    links.new(diffuse_bsdf.outputs[0], material_output.inputs["Surface"])

    return mesh_object


def generate_faces(image_width, image_height):
    indices = np.arange(image_width * image_height).reshape(image_height, image_width)
    lower_left = indices[:-1, :-1].ravel()
    lower_right = indices[:-1, 1:].ravel()
    upper_left = indices[1:, :-1].ravel()
    upper_right = indices[1:, 1:].ravel()
    faces = np.column_stack(
        (
            np.column_stack((lower_left, lower_right, upper_left)),
            np.column_stack((lower_right, upper_right, upper_left)),
        )
    ).reshape(-1, 3)
    return faces


def setup_mesh(image_path, depth_anything_model):
    # 加载图像和 disparity
    image = cv2.imread(image_path)

    disparity = depth_anything_model.infer_image(image[:, :, ::-1], 518).astype(
        np.float32
    )

    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())

    H, W = image.shape[:2]

    print(disparity.shape, image.shape)

    if H > W:
        image = imutils.resize(image, width=512)
        disparity = imutils.resize(disparity, width=512)
    elif H < W:
        image = imutils.resize(image, height=512)
        disparity = imutils.resize(disparity, height=512)
    else:
        image = imutils.resize(image, height=512, width=512)
        disparity = imutils.resize(disparity, height=512, width=512)

    print(disparity.shape, image.shape)

    disparity = np.clip(disparity, 0.01, 1)

    disparity = scipy.ndimage.gaussian_filter(disparity, sigma=2)

    print(disparity.shape, image.shape)

    cv2.imwrite("image.png", image)

    image_height, image_width = image.shape[:2]

    # 相机内参
    focal = 0.58 * image_width
    cx = image_width / 2
    cy = image_height / 2
    fx = focal
    fy = focal

    camera_setup = {
        "focal": focal,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_height": image_height,
        "image_width": image_width,
    }

    # 计算深度
    depth = 1.0 / disparity

    depth = depth * (np.random.random() * 0.6 + 0.7) + np.random.random() * 0.2

    print(depth.shape, image.shape)

    subdivision = 2

    # 生成像素网格
    u_coords, v_coords = np.meshgrid(
        np.arange(image_width * subdivision), np.arange(image_height * subdivision)
    )

    # 将像素坐标转换为相机坐标系
    Y = imutils.resize(depth, width=subdivision * 512)
    X = (u_coords / subdivision - cx) * Y / fx
    Z = (v_coords / subdivision - cy) * Y / fy * -1

    # 合并为点云
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = (
        imutils.resize(image, width=subdivision * 512).reshape(-1, 3)[:, ::-1] / 255.0
    )
    

    # 生成三角形网格
    faces = generate_faces(image_width * subdivision, image_height * subdivision)
    # 创建 Blender 中的网格
    mesh_object = create_mesh(points, faces, colors)

    return mesh_object, camera_setup


def setup_gpu_rendering():
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    # 获取 Eevee 渲染设置
    eevee = bpy.context.scene.eevee

    # 启用一些高级渲染设置
    eevee.use_ssr = True  # 启用屏幕空间反射

    print("Eevee render settings configured to use GPU.")


def setup_scene(camera_setup, mesh_object, output_path):
    # 清理现有场景（删除所有物体、灯光、相机等）

    # 确保当前处于对象模式
    if bpy.context.object:
        bpy.ops.object.mode_set(mode="OBJECT")

    # 将生成的网格添加到场景中
    # bpy.context.collection.objects.link(mesh_object)

    for _ in range(np.random.randint(2, 6)):
        # 创建并设置光源（这里使用点光源）
        light_data = bpy.data.lights.new(name="PointLight", type="POINT")
        light_data.energy = 400 + 1600 * np.random.rand()  # 光源强度 (1000~3000)
        light_data.color = (
            np.random.rand() * 0.5 + 0.5 if np.random.rand() > 0.5 else 0.5,
            np.random.rand() * 0.5 + 0.5 if np.random.rand() > 0.5 else 0.5,
            np.random.rand() * 0.5 + 0.5 if np.random.rand() > 0.5 else 0.5,
        )  # 设置光源颜色

        light_obj = bpy.data.objects.new(
            name="PointLightObject", object_data=light_data
        )
        light_obj.location = (
            (
                np.random.rand() * 3
                if np.random.rand() > 0.5
                else -1 * np.random.rand() * 2
            ),  # -4~4
            np.random.randint(1, 3),  # [1, 2, 3]
            (
                np.random.rand() * 2
                if np.random.rand() > 0.5
                else -1 * np.random.rand() * 2
            ),  # -3~3
        )  # 设置光源位置
        bpy.context.scene.collection.objects.link(light_obj)
        print(light_data.color, light_obj.location)

    # 设置环境光（使用节点设置背景）
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes["Background"]
    bg_node.inputs[1].default_value = 1.6 + np.random.rand()  # 增加环境光亮度

    # 创建相机
    camera_data = bpy.data.cameras.new(name="Camera")

    camera_data.lens = camera_setup["focal"] / 10

    camera_obj = bpy.data.objects.new(name="CameraObject", object_data=camera_data)

    # 获取相机的传感器宽度（单位：毫米）
    sensor_width = camera_obj.data.sensor_width
    print(f"Sensor Width: {sensor_width} mm")

    sensor_fit = camera_obj.data.sensor_fit
    print(f"Sensor Fit: {sensor_fit}")

    # 假设图像宽度为像素单位
    image_width_pixels = camera_setup["image_width"]

    # 计算焦距（毫米）
    focal_mm = (camera_setup["focal"] / image_width_pixels) * sensor_width

    camera_data.lens = focal_mm

    # 相机位置和旋转
    camera_obj.location = (0, 0, 0)  # 确保相机在模型前方
    camera_obj.rotation_euler = (math.radians(90), 0, 0)  # 将相机对准模型
    bpy.context.scene.collection.objects.link(camera_obj)

    # 设置场景中的相机
    bpy.context.scene.camera = camera_obj

    # 设置渲染引擎为 Cycles
    setup_gpu_rendering()

    # 渲染设置
    bpy.context.scene.render.resolution_x = camera_setup[
        "image_width"
    ]  # 设置渲染图像的分辨率宽度
    bpy.context.scene.render.resolution_y = camera_setup[
        "image_height"
    ]  # 设置渲染图像的分辨率高度
    bpy.context.scene.render.film_transparent = True  # 设置背景透明，方便后期处理

    # 设置采样数（可以调整）
    bpy.context.scene.cycles.samples = (
        32  # 这里设置了一个较低的采样数，实际使用时可以调整
    )

    # 设置输出图像路径
    bpy.context.scene.render.filepath = output_path

    # 渲染当前场景并保存图像
    bpy.ops.render.render(write_still=True)

    # 保存渲染结果
    bpy.data.images["Render Result"].save_render(filepath=output_path)  # 保存渲染结果
    print(f"Render saved to {output_path}")


if __name__ == "__main__":

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    import torch

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    depth_anything = DepthAnythingV2(**model_configs["vitl"]).half()
    depth_anything.load_state_dict(
        torch.load(f"checkpoints/depth_anything_v2_vitl.pth", map_location="cpu")
    )

    depth_anything = depth_anything.to(DEVICE).eval()

    # 输入参数
    image_path = "634.jpg"
    rendered_image_path = "634_relight.png"

    mesh_object, camera_setup = setup_mesh(image_path, depth_anything)

    setup_scene(camera_setup, mesh_object, rendered_image_path)
