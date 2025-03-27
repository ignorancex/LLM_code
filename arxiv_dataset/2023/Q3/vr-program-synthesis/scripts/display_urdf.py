import tempfile
import time
from argparse import ArgumentParser

import pybullet as pb
import pybullet_data

from vr_program_synthesis.utils import resolve_urdf


def add_link_coordinate_system(obj_id: int, link_id: int, axis_length=0.05):
    pb.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], lineWidth=3, parentObjectUniqueId=obj_id,
                       parentLinkIndex=link_id)
    pb.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], lineWidth=3, parentObjectUniqueId=obj_id,
                       parentLinkIndex=link_id)
    pb.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], lineWidth=3, parentObjectUniqueId=obj_id,
                       parentLinkIndex=link_id)


def main(args):
    physicsClient = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    resolved_urdf_str = resolve_urdf(args.urdf_filepath)
    tf = tempfile.NamedTemporaryFile(suffix=".urdf")
    with open(tf.name, "w") as fp:
        fp.write(resolved_urdf_str)
    obj_id = pb.loadURDF(tf.name)
    if args.display_frames:
        for link_id in range(pb.getNumJoints(obj_id)):
            joint_info = pb.getJointInfo(obj_id, link_id)
            joint_frame_position = joint_info[14]
            joint_frame_orientation = joint_info[15]
            link_name = joint_info[12]
            # if link_name.decode("utf-8") == "default_point":
            #     print(f"Drawing link frame for {link_name}: {joint_frame_position}, {joint_frame_orientation}")
            add_link_coordinate_system(obj_id, link_id)
    while True:
        pb.stepSimulation()
        time.sleep(1/240)
    pb.disconnect()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("urdf_filepath", type=str)
    parser.add_argument("--display_frames", action="store_true")
    main(parser.parse_args())
