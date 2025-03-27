import gym
import pybullet_multigoal_gym as pmg
import envs.fetch as fetch_env
import envs.hand as hand_env
import envs.kuka as kuka_env
from .utils import goal_distance, goal_distance_obs

Robotics_envs_id = [
    'FetchReach-v1',
    'FetchPush-v1',
    'FetchSlide-v1',
    'FetchPickAndPlace-v1',
    'FetchPushNew-v1',
    'FetchCurling-v1',
    'FetchPushObstacle-v1',
    'FetchPickObstacle-v1',
    'FetchPushNoObstacle-v1',
    'FetchPickNoObstacle-v1',
    'FetchPushLabyrinth-v1',
    'FetchPickAndThrow-v1',
    'FetchPickAndSort-v1',
    'HandManipulateBlock-v0',
    'HandManipulateBlockRotateY-v0',
    'HandManipulateEgg-v0',
    'HandManipulatePen-v0',
    'HandReach-v0',
    'PMGPush-v0'
]
Kuka_envs_id = [
    'KukaReach-v1',
    'KukaPickAndPlaceObstacle-v1',
    'KukaPickAndPlaceObstacle-v2',
    'KukaPickNoObstacle-v1',
    'KukaPickNoObstacle-v2',
    'KukaPickThrow-v1',
    'KukaPushLabyrinth-v1',
    'KukaPushLabyrinth-v2',
    'KukaPushSlide-v1',
    'KukaPushNew-v1'
]


def make_env(args):
    assert (args.env in Robotics_envs_id or args.env in Kuka_envs_id)
    if args.env[:4] == 'Kuka':
        return kuka_env.make_env(args)
    elif args.env[:5] == 'Fetch':
        return fetch_env.make_env(args)
    elif args.env[:4] == 'Hand':
        return hand_env.make_env(args)
    else:
        camera_setup = [{
            'cameraEyePosition': [-0.9, -0.0, 0.4],
            'cameraTargetPosition': [-0.45, -0.0, 0.0],
            'cameraUpVector': [0, 0, 1],
            'render_width': 224,
            'render_height': 224},
            {
            'cameraEyePosition': [-1.0, -0.25, 0.6],
            'cameraTargetPosition': [-0.6, -0.05, 0.2],
            'cameraUpVector': [0, 0, 1],
            'render_width': 224,
            'render_height': 224
            },]
        return pmg.make_env(task='push',gripper='parallel_jaw',num_block=1,render=False, binary_reward=True,max_episode_steps=50, image_observation=False, depth_image=False, goal_image=False, visualize_target=False, camera_setup=camera_setup, observation_cam_id=[0], goal_cam_id=1)




def clip_return_range(args):
    gamma_sum = 1.0 / (1.0 - args.gamma)
    return {
        'FetchReach-v1': (-gamma_sum, 0.0),
        'FetchPush-v1': (-gamma_sum, 0.0),
        'FetchPushNew-v1': (-gamma_sum, 0.0),
        'FetchSlide-v1': (-gamma_sum, 0.0),
        'FetchPickAndPlace-v1': (-gamma_sum, 0.0),
        'FetchPickObstacle-v1': (-gamma_sum, 0.0),
        'FetchPickNoObstacle-v1': (-gamma_sum, 0.0),
        'FetchPushLabyrinth-v1': (-gamma_sum, 0.0),
        'FetchPickAndThrow-v1': (-gamma_sum, 0.0),
        'FetchPickAndSort-v1': (-gamma_sum, 0.0),
        'HandManipulateBlock-v0': (-gamma_sum, 0.0),
        'HandManipulateBlockRotateY-v0': (-gamma_sum, 0.0),
        'HandManipulateEgg-v0': (-gamma_sum, 0.0),
        'HandManipulatePen-v0': (-gamma_sum, 0.0),
        'HandReach-v0': (-gamma_sum, 0.0),
        'KukaReach-v1': (-gamma_sum, 0.0),
        'PMGPush-v0': (-gamma_sum, 0.0),
        'KukaPickAndPlaceObstacle-v1': (-gamma_sum, 0.0),
        'KukaPickAndPlaceObstacle-v2': (-gamma_sum, 0.0),
        'KukaPickNoObstacle-v1': (-gamma_sum, 0.0),
        'KukaPickNoObstacle-v2': (-gamma_sum, 0.0),
        'KukaPickThrow-v1': (-gamma_sum, 0.0),
        'KukaPushLabyrinth-v1': (-gamma_sum, 0.0),
        'KukaPushLabyrinth-v2': (-gamma_sum, 0.0),
        'KukaPushSlide-v1': (-gamma_sum, 0.0),
        'KukaPushNew-v1': (-gamma_sum, 0.0)
    }[args.env]
