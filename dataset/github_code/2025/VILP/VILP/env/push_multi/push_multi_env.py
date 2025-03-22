import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions

def angle_to_quat(angle):
    return np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class PushMultiEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size_h=96,
            render_size_w=160,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_scale = 5
        self.mode = 'rgb_array'

        self.render_size = [render_size_w, render_size_h]
        self.window_size = [render_size_w*self.window_scale, render_size_h*self.window_scale]
        ws = self.window_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy


        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,self.render_size[1],self.render_size[0]),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=np.array([0,0,]),
                high=np.array([ws[0],ws[1]]),
                shape=(2,),
                dtype=np.float32
            ),
        })
        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws[0],ws[1]], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
    
    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block1.center_of_gravity = self.block_cog
            self.block2.center_of_gravity = self.block_cog
            self.block3.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            # set three blocks
            state1 = np.array([
                rs.randint(100, 700), rs.randint(100, 380),
                rs.randn() * 2 * np.pi - np.pi
            ])
            state2 = np.array([
                rs.randint(100, 700), rs.randint(100, 380),
                rs.randn() * 2 * np.pi - np.pi
            ])
            state3 = np.array([
                rs.randint(100, 700), rs.randint(100, 380),
                rs.randn() * 2 * np.pi - np.pi
            ])
            pos = np.array([rs.randint(50, 750), rs.randint(50, 430)])
        self._set_state(state1, state2, state3,pos)

        observation = self._get_obs()

        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)


        reward = 0
        done = False
        tolerance = 10
        if (abs(self.block1.position[1] - 
               self.block2.position[1]) < tolerance) and (abs(self.block1.position[1] - 
                self.block3.position[1]) < tolerance):
            done = True
            reward = 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info
    

    def render(self,mode='rgb_array'):

        if self.render_cache is None:
            self._get_obs()
        return self.render_cache

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        mode = self.mode
        img = self._render_frame(mode=mode)

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {

            'image': img_obs,
            'agent_pos': agent_pos,
        }
        self.render_cache = img
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):

        info = {
            'agent_pos': np.array(self.agent.position),}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size[0], self.window_size[1]))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)



        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size[0], self.render_size[1]))
        '''
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / self.window_scale).astype(np.int32)
                marker_size = int(8/96*self.render_size[0])
                thickness = int(1/96*self.render_size[0])
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        '''
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state1, state2, state3,pos):
        if isinstance(state1, np.ndarray):
            state1 = state1.tolist()
        if isinstance(state2, np.ndarray):
            state2 = state2.tolist()
        if isinstance(state3, np.ndarray):
            state3 = state3.tolist()
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()
        pos_agent = pos
        pos_block1 = state1[:2]
        rot_block1 = state1[2]
        pos_block2 = state2[:2]
        rot_block2 = state2[2]
        pos_block3 = state3[:2]
        rot_block3 = state3[2]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block1.position = pos_block1
            self.block1.angle = rot_block1
            self.block2.position = pos_block2
            self.block2.angle = rot_block2
            self.block3.position = pos_block3
            self.block3.angle = rot_block3
        else:
            self.block1.position = pos_block1
            self.block2.position = pos_block2
            self.block3.position = pos_block3
            self.block1.angle = rot_block1
            self.block2.angle = rot_block2
            self.block3.angle = rot_block3

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 475), (5, 5), 2),  # left wall
            self._add_segment((5, 5), (795, 5), 2),  # top wall
            self._add_segment((795, 5), (795, 475), 2),  # right wall
            self._add_segment((5, 475), (795, 475), 2)  # bottom wall
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 20)
        self.block1 = self.add_box((256, 256), 85, 85)
        self.block2 = self.add_box((256, 256), 85, 85)
        self.block3 = self.add_box((256, 256), 85, 85)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        #self.success_threshold = 0.95    # 95% coverage.
        self.success_threshold = 0.9    # 90% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        #shape.color = pygame.Color('Purple')
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
