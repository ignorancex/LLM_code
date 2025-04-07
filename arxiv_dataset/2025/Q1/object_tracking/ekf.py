import sys
from config import sessions

# @profile
def process(session):
    import math
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use('Agg') # apparantly needed to avoid memory leak
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Stop spam from tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # dont use gpu because it leads to jit compilation error
    import tensorflow as tf
    # from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation
    from config import alpha, groundtruth_w, predifined_classes, img_shape, trials
    from config import all_camera_names as all_camera_names_gt # all camera names that could exist
    from typing import Dict
    import tempfile
    import yaml
    from functools import lru_cache

    @tf.function
    def project (pInX, cInX, alpha, u0, v0, k1=0, k2=0):
        """Implements the camera model also used in OpenCV limited to one(two?) coefficient of radial distortion.
        If a point is at pInX (4 vector, in some coordinate system X) and the camera is at cInX (pose, 4*4 matrix)
        and the camera has focal length alpha, image center u0, v0 and radial distortion k1, k2 then the point will be seen
        in the image at the returned 2D position. If a point is behind the camera
        NaN is returned. This function works with arbitrary many leading batch 
        dimensions in pInX and/or cInX.
        (Function originally from Udo Freese's 3DBV Lecture)"""
        pInC = tf.einsum ('...ij,...j->...i', tf.linalg.inv(cInX), pInX)
        z = pInC[...,2]
        # if z<=0 the point is behind the camera an we propagate NaN
        x = tf.where(z>0, pInC[...,0]/z, math.nan) 
        y = tf.where(z>0, pInC[...,1]/z, math.nan)
        r2 = x**2 + y**2
        factor = (1 + k1*r2 + k2*r2**2)*alpha
        res = tf.stack([factor*x+u0, factor*y+v0], axis=-1)
        return res

    @tf.function
    def camera_model(session, trial, points_in_W, camera_name, focal_length, imageFormat):
        '''given 3d world coordinates, calculate expected 2d pixel position'''
        return project(
                tf.concat((points_in_W, tf.constant([1], dtype=tf.float64)), axis=0)[tf.newaxis,:], # weird reshaping of point in 3d space
                c_in_W(session, trial, camera_name), # pose of camera in 3d space
                focal_length,
                imageFormat[1]/2, # image width
                imageFormat[0]/2 # image heigth
            )

    @lru_cache(maxsize=6) 
    def c_in_W(session, trial, camera_name):
        params_this_cam = get_params_for_cam(session, trial, camera_name)
        '''matrix that describes the camera pose in world coordinates'''
        FInTminus = np.array([[1, 0, 0, params_this_cam[0]],
                            [0, 0, 1, params_this_cam[1]],
                            [0,-1, 0, params_this_cam[2]],
                            [0, 0, 0, 1]])
        TminusInPan = Rotation.from_rotvec([0, params_this_cam[3], 0]).as_matrix()
        TminusInPan = np.pad(TminusInPan, (0, 1), 'constant', constant_values=(0))
        TminusInPan[3, 3] = 1
        
        PanInTilt = Rotation.from_rotvec([params_this_cam[4], 0, 0]).as_matrix()
        PanInTilt = np.pad(PanInTilt, (0, 1), 'constant', constant_values=(0))
        PanInTilt[3, 3] = 1
        
        TiltInTplus = Rotation.from_rotvec([0, 0, params_this_cam[5]]).as_matrix()
        TiltInTplus = np.pad(TiltInTplus, (0, 1), 'constant', constant_values=(0))
        TiltInTplus[3, 3] = 1
        
        # multiply trainsformation matrices to get cam in world transformation matrix
        ret = FInTminus @ TminusInPan @ PanInTilt @ TiltInTplus
        return ret

    def dist_point_to_line_np(x0, x1, x2):
        '''x0 is point and x1, x2 are points defining the line.
        https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html'''
        cross = np.cross(x2-x1, x1-x0)
        norm1 = np.linalg.norm(cross)
        norm2 = np.linalg.norm(x2-x1)
        ret = np.divide(norm1, norm2)
        return ret

    class Detection_line: # a line in 3D space that represents all possible 3D points of a cameras 2D yolo detection
        def __init__(self, pixel_coords, cam_coords, other_point, tableware_class, camera_name, yolo_conf):
            self.pixel_coords = np.array(pixel_coords)
            self.cam_coords = np.array(cam_coords)
            self.other_point = np.array(other_point)
            self.direction = self.other_point - self.cam_coords
            self.direction = self.direction / np.linalg.norm(self.direction)
            self.tableware_class = tableware_class
            self.camera_name = camera_name
            self.yolo_conf = yolo_conf
        def __str__(self):
            return f'\n{self.camera_name}, {self.tableware_class}, direction:{[round(x, 2) for x in self.direction.tolist()]}'
        def __repr__(self):
            return self.__str__()

    class Detection_point: # a point in 3D space defined by a number of (nearly) intersecting detection_lines
        def __init__(self, coords, detection_lines):
            self.coords = coords
            self.detection_lines = detection_lines
            self.tableware_class = None if len(detection_lines)==0 else detection_lines[0].tableware_class
            for detection_line in self.detection_lines:
                assert detection_line.tableware_class == self.tableware_class, f'found conflicting classes {detection_line.tableware_class} and {self.tableware_class}'
            self.camera_names = [line.camera_name for line in self.detection_lines]
            assert len(self.camera_names) == len(set(self.camera_names)), f'got same camera multiple times in {self.camera_names}'
            self.error = sum([dist_point_to_line_np(self.coords, line.cam_coords, line.other_point) for line in self.detection_lines])
        def __str__(self):
            return f'\n{self.tableware_class} at {[round(x, 2) for x in self.coords.tolist()]}: {self.camera_names}'
        def __repr__(self):
            return self.__str__()
        
    @lru_cache(maxsize=6) 
    def get_params_for_cam(session, trial, camera_name=None):
        path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
        with open(f'{path_prefix}mocap.objecttracking.cameraposes.yaml') as f:
                params_dict = yaml.safe_load(f)
        # print(params_dict.keys())
        if not (camera_name is None):
            x = params_dict[camera_name]
            return [x['x']] + [x['y']] + [x['z']] + [x['a_pan']] + [x['a_tilt']] + [x['a_roll']]
        else:
            return params_dict

    @lru_cache(maxsize=6) 
    def get_detections(session, trial):
        yolo_detections = {}
        print(f'downloading detections for session {session}, trial {trial}')
        path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
        for camera_name in all_camera_names:
            with open(f'{path_prefix}mocap.objecttracking.{camera_name}.yolodetections.yaml') as f:
                yolo_detections[camera_name] = yaml.safe_load(f)
        return yolo_detections
            
    line_scale = 5 # num meters the points defining a detection line are apart

    def get_detections_of_frame(session, trial, frame_num):
        detection_lines_this_frame = []
        for camera_name in all_camera_names:
            cam_params = get_params_for_cam(session, trial, camera_name)
            for detection in get_detections(session, trial)[camera_name][str(frame_num)]:
                x = (detection[1][0]-(img_shape[1]/2)) / alpha
                y = (detection[1][1]-(img_shape[0]/2)) / alpha
                detection3d_in_cam_coords = np.array([x*line_scale, y*line_scale, 1*line_scale, 1])
                # calc a point in that lies on the line between camera and detected object
                detection_point3d_in_world = (c_in_W(session, trial, camera_name) @ detection3d_in_cam_coords)[:3]
                line_obj = Detection_line((detection[1][0], detection[1][1]), cam_params[:3], detection_point3d_in_world, detection[0], camera_name, yolo_conf=detection[2])
                detection_lines_this_frame.append(line_obj)
        return detection_lines_this_frame

    class ExtendedKalmanFilter:
        def __init__(self, measurement_noise_sigma: np.double, initial_guess, tableware_class, frame, session, trial):
            self.mu_x = tf.constant(initial_guess, dtype=tf.float64) # The filter's belief on the current state
            self.cov_x = tf.constant(np.identity(3, dtype=np.double), dtype=tf.float64)  # The state covariance
            self.measurement_noise_sigma = measurement_noise_sigma
            self.tableware_class = tableware_class # class/name of tracked object. eg plate
            self.C_t = {} # cache some matrices to save on runtime
            self.last_z = {} # cache last measurement by camera. If it does not change, we save redundant measurment steps
            self.last_frame = frame # the last frame/ timestep we performed a measurement step on
            self.last_actual_frame = frame # when did we last not only accept a measurement but also actually use it
            self.trajectory = [] # This is where elements of [frame_num, position, covariance] are appended
            self.num_measurements = 0 # Number of measurement steps performed to quickly asses if this kf is relevant
            self.first_frame = frame # Start of this kf's life
            self.dead = False # Kf wont be updated anymore once this flag is set
            self.residuum_rolling_avg = 100 # Keep track of how well our kf believed position fits in with new measuremts
            self.queued_measurements = [] # dont process possible outlieres until we have no better option
            self.session = session
            self.trial = trial
            self.stationary = False # Flag to save redundant measurment steps
            
        @tf.function
        def measurement_model(self, point_in_W, camera_name):
            """
            maps points in 3D World to pixelpos in respective cam
            """
            # cam_param = tf.constant(get_params_for_cam(self.session, self.trial, camera_name), dtype=tf.float64)
            return camera_model(self.session, self.trial, point_in_W, camera_name, alpha, img_shape)
        
        def measurement_model_jacobian(self, point_in_W, camera_name):
            """ Jacobian of g (measurement_model) with respect to point_in_W """
            # cam_param = tf.constant(get_params_for_cam(self.session, self.trial, camera_name), dtype=tf.float64)
            point_in_W_var = tf.Variable(point_in_W)
            with tf.GradientTape() as tape:
                y = camera_model(self.session, self.trial, points_in_W=point_in_W_var, camera_name=camera_name, focal_length=alpha, imageFormat=img_shape)
            jac = tape.jacobian(y, point_in_W_var)
            self.C_t[camera_name] = {'C_t': jac[0], 'mu_x': self.mu_x}
            return jac[0]

        def measurement(self, _z: np.ndarray, camera_name, frame_num, yolo_conf, recursive=False):
            # check if can skip this measuremt
            if camera_name in self.last_z:
                # if we have not moved much in image coordinates then skip actual measurement step
                self.stationary = np.linalg.norm(_z - self.last_z[camera_name]) < 10 and self.residuum_rolling_avg < 10
                if self.stationary: # dont waste time on same measurement again and again
                    self.last_frame = frame_num
                    self.num_measurements += 1
                    return
            self.last_z[camera_name] = _z
            # residuum is diff between exptected measuremt given our believed position and actual measuremt
            residuum = _z - self.measurement_model(self.mu_x, camera_name)
            residuum_norm = tf.math.reduce_euclidean_norm(residuum)
            self.residuum_rolling_avg = 0.9 * self.residuum_rolling_avg + 0.1 * residuum_norm
            # rule out inplausible measuremts if possible
            if not recursive and residuum_norm > 3 * self.residuum_rolling_avg:
                self.queued_measurements.append({'_z': _z, 'camera_name': camera_name, 'frame_num': frame_num, 'yolo_conf': yolo_conf})
                if len(self.queued_measurements) > 5: # 5 bad measurements cant be explained by near objects of same class
                    for m in self.queued_measurements:
                        self.measurement(m['_z'], m['camera_name'], m['frame_num'], m['yolo_conf'], recursive=True)
                return

            if not recursive:
                self.queued_measurements = [] # We have a measurement with low residuum so we do not need to worry about the queue of inplausible measuremts

            if camera_name in self.C_t and tf.math.reduce_euclidean_norm(self.C_t[camera_name]['mu_x']-self.mu_x) < 1:
                C_t = self.C_t[camera_name]['C_t']
            else:
                C_t = self.measurement_model_jacobian(self.mu_x, camera_name)
            if tf.reduce_sum(C_t).numpy() == 0: # The believe state is probably behind the camera
                tf.dead = True
                return
                
            yolo_factor = (2 - yolo_conf) ** 10 # modify measuremt noise with knowledge of yolo confidence
            # Standard Extended Kalman Filter code from here
            cov_delta_t = (np.identity(2, dtype=np.double) * self.measurement_noise_sigma * yolo_factor) ** 2
            bracket_term = C_t @ self.cov_x @ tf.transpose(C_t) + cov_delta_t
            K_t = self.cov_x @ tf.transpose(C_t) @ tf.linalg.inv(bracket_term)
            self.mu_x = self.mu_x + tf.transpose(K_t @ tf.transpose(residuum))[0]
            self.cov_x = self.cov_x - K_t @ C_t @ self.cov_x
            self.last_frame = frame_num
            self.last_actual_frame = frame_num
            self.num_measurements += 1
            return
            
        # called once every timestep / frame. covariance increases as the object might move with time
        def dynamic(self, frame_num, is_last_frame):
            if frame_num - self.last_frame > 100:
                self.dead = True
            dist_to_origin = abs(tf.math.reduce_euclidean_norm(self.mu_x - groundtruth_w['origin']))
            if (not self.dead) and self.residuum_rolling_avg < 100 and dist_to_origin < 5: # dont write data if its obviously wrong
                self.trajectory.append([frame_num, self.mu_x.numpy().tolist(), self.cov_x.numpy().tolist()])
                if (not self.stationary) or self.last_actual_frame == frame_num: # if we became stationary during frame, we still need to update covariance
                    self.cov_x += tf.constant(np.identity(3, dtype=np.double) * 0.1)

    # This takes a list of 3d lines, each derived from a 2d yolo detection in one camera
    # returns intersection points of these lines
    # Note that it should only be called with detections of one tableware_class at a time to reduce runtime
    def get_intersections(detections):
        eps = 0.1 # allowed distance in m
        minimum_angle = np.pi/8 # if two cameras see an object in nearly the same direction, we wont get an accurate 3d position
        max_dist_to_cam = 5 # meters
        detection_points = []
        for camera_name in all_camera_names:
            det_this_cam = [d for d in detections if d.camera_name == camera_name]
            det_other_cams = [d for d in detections if d.camera_name != camera_name]
            if len(det_other_cams) == 0:
                continue
            for det in det_this_cam:
                dist_to_cam = 0 # we will find points on the line by increasing the distance to the camera
                while dist_to_cam < max_dist_to_cam: # check if line has (near) intersections within first max_dist_to_cam meters
                    # calc dist to every line in lines_of_current_class_and_other_cams. This is done in parallel because this is a runtime bottleneck 
                    point = det.cam_coords + det.direction * dist_to_cam
                    x1 = np.array([det.cam_coords for det in det_other_cams])
                    x2 = np.array([det.other_point for det in det_other_cams])
                    cross = np.cross(x2-x1, x1-point, axis=1)
                    norm1 = np.linalg.norm(cross, axis=1)
                    norm2 = np.linalg.norm(x2-x1, axis=1)
                    dists_to_other_det = np.divide(norm1, norm2)
                    
                    closest_dist_to_other_det = min(dists_to_other_det)
                    success = False
                    for dist_to_other_det, other_det in zip(dists_to_other_det, det_other_cams):
                        if dist_to_other_det < eps: # we have at least two close lines, which means we have detected the 3d position of an object
                            angle = np.arccos(np.clip(np.dot(det.direction, other_det.direction), -1.0, 1.0))
                            # intersecting lines should have a minimum angle as near parallel lines cant determine a point accurately in all dimensions
                            if angle > minimum_angle:
                                success = True
                                other_lines_this_detection = [l for i, l in enumerate(det_other_cams) if dists_to_other_det[i] < eps]
                                
                                # ensure we only have one line per camera per class (this happens when one cam has two detections very close to each other)
                                other_cams = set()
                                other_lines_to_delete = []
                                for other_line in other_lines_this_detection:
                                    if other_line.camera_name in other_cams:
                                        other_lines_to_delete.append(other_line)
                                    other_cams.add(other_line.camera_name)
                                for other_line in other_lines_to_delete:
                                    other_lines_this_detection.remove(other_line)
                                    
                                if len(other_lines_this_detection) > 1 or predifined_classes[det.tableware_class]: # we need at least 3 lines in total, because 2 lines are ambigous if the tableware_class exists multiple times
                                    point_obj = Detection_point(point, [det] + other_lines_this_detection)
                                    detection_points.append(point_obj)
                                dist_to_cam += eps/2
                    if not success:
                            dist_to_cam += closest_dist_to_other_det # if lines are at least x meters away, move x meters along our line
        
        # remove duplicate points
        points = []
        for detection_point in detection_points:
            dists = [np.linalg.norm(other_point.coords - detection_point.coords) for other_point in points]
            if len(dists) == 0 or min(dists) > eps * 5:
                points.append(detection_point)
        return points

    for trial in trials:
        print(f'processing session {session} trial {trial}')
        path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
        try:
            all_camera_names = [name for name in get_params_for_cam(session, trial) if name in all_camera_names_gt] # find out which cameras are usable
        except:
            print('could not download cameraposes, skipping trial')
            continue
        kf_by_class = dict(zip(predifined_classes.keys(), [[] for _ in predifined_classes])) # a list of kalmanfilters for every tableware class
        try:
            num_frames = min([len(get_detections(session, trial)[camera_name]) for camera_name in all_camera_names])
        except:
            print('could not download yolo detections, skipping trial')
            continue
        # num_frames = 500 # Just for testing
        for frame_num in range(0, num_frames):
            # if frame_num % 100 == 0:
            #     print('frame:', frame_num)
            #     print('num kfs alive', sum([len([kf for kf in kf_by_class[c] if not kf.dead]) for c in predifined_classes]))
            #     print('num kfs total', sum([len(kf_by_class[c]) for c in predifined_classes]))
            # We take 2d YOLO detections and interpret them as lines in 3d space
            detections = get_detections_of_frame(session, trial, frame_num)
            detections = [d for d in detections if d.yolo_conf > 0.5]
            for tableware_class in predifined_classes:
                if frame_num % 100 == 0: # every once in a while delete failed kfs
                    kf_by_class[tableware_class] = [kf for kf in kf_by_class[tableware_class] if kf.num_measurements > 30 or (not kf.dead)]
                det_of_class = [d for d in detections if d.tableware_class == tableware_class]
                unused_det_of_class = det_of_class # used lines will be removed from this list
                kfs = [kf for kf in kf_by_class[tableware_class] if not kf.dead] # all active kalman filter instances with tableware_class
                for camera_name in all_camera_names:
                    det_of_class_and_cam = [d for d in det_of_class if d.camera_name==camera_name]
                    for kf in kfs:
                        line_dists = [dist_point_to_line_np(kf.mu_x.numpy(), line.cam_coords, line.other_point) for line in det_of_class_and_cam]
                        if len(line_dists) > 0 and ( min(line_dists) < 0.3 or predifined_classes[tableware_class] ):
                            line = det_of_class_and_cam[line_dists.index(min(line_dists))]
                            if line in unused_det_of_class:
                                unused_det_of_class.remove(line)
                            else: # some other kf has already used this detection line. This should not be and we probably need to retire one of the kfs
                                closest_kf = None
                                closest_dist = np.inf
                                for other_kf in kfs:
                                    dist = np.linalg.norm(kf.mu_x.numpy() - other_kf.mu_x.numpy())
                                    if dist < closest_dist and other_kf != kf:
                                        closest_kf = other_kf
                                        closest_dist = dist
                                if closest_dist < 0.01:
                                    # We disable one of the competing kfs from making future measurements
                                    if len(kf.trajectory) > len(closest_kf.trajectory):
                                        closest_kf.dead = True
                                    else:
                                        kf.dead = True
                            # With the detection lines of given tableware_class and camera, take the line that is closest in 3d space to our kf's believed position and do measurement step
                            kf.measurement(line.pixel_coords, line.camera_name, frame_num, line.yolo_conf)     
                # with all the lines that where not used as measurements by any kf, find line intersections for possible newe kf
                new_points = get_intersections(unused_det_of_class)
                for point in new_points:
                    if len(kfs) > 0:
                        min_dist = min([np.linalg.norm(kf.mu_x.numpy() - point.coords) for kf in kfs])
                    else:
                        min_dist = np.inf
                    thrs = 0.01 if (frame_num%50==0 or frame_num in [1,2,3]) else 0.25
                    # only try to establish a new kf if out point is relativly far away from objects of same tableware_class, as the new kf would likely just track the same object, wasting resources.
                    # to be able to track close objects of same tableware_class anyway, we drop the threshold of minimal distance for new kf every 50 frames.
                    if min_dist > thrs:
                        kf = ExtendedKalmanFilter(20, tf.constant(point.coords.tolist(), dtype=tf.float64), tableware_class, frame_num, session, trial)
                        kf_by_class[tableware_class].append(kf)
                    point = None # memory leak fix?
                for kf in [kf for kf in kf_by_class[tableware_class] if not kf.dead]:
                    # This is called once for every frame and kf. This is where kfs that are not dead write their trajectories
                    kf.dynamic(frame_num, frame_num==num_frames-1)
        
        # collect all relevant trajectories of kfs
        trajectories = []
        for tableware_class in kf_by_class:
            for kf in kf_by_class[tableware_class]:
                if kf.num_measurements > 30 and len(kf.trajectory) > 30:
                    trajectories.append((tableware_class, kf.trajectory))
        
        # post processing: freeze objects in place if they are not moving far enough in time window
        window_size = 30
        smoothing_distance = 0.03 # distance in meters an object has to move to also move in smoothed data
        smoothed_trajectories = []
        for tableware_class, trajectory in trajectories:
            smoothed_trajectory = trajectory[:window_size]
            last_pos = np.mean([x[1] for x in trajectory[0:window_size]], axis=0).tolist()
            for i in range(window_size, len(trajectory)-window_size):
                avg_past = np.mean([x[1] for x in trajectory[i-window_size:i]], axis=0)
                avg_future = np.mean([x[1] for x in trajectory[i:i+window_size]], axis=0)
                if np.linalg.norm(avg_past - avg_future) < smoothing_distance:
                    # copy position from last timestep, set covariance to 0
                    smoothed_trajectory.append([trajectory[i][0], [last_pos[0], last_pos[1], last_pos[2]],
                                                [[0,0,0], [0,0,0], [0,0,0]]])
                    if i==100:
                        print('smoothed to:', smoothed_trajectory[-1])
                else:
                    smoothed_trajectory.append(trajectory[i])
                    last_pos = np.array(trajectory[i][1]).tolist()
            smoothed_trajectories.append([tableware_class, smoothed_trajectory])
            
        # plot all trajectories in png file to quickly visualize results
        plt.clf()
        table = groundtruth_w['table'].numpy().tolist()
        table.append(table[0])
        plt.plot([p[0] for p in table], [p[1] for p in table])
        counter = groundtruth_w['counter'].numpy().tolist()
        counter.append(counter[0])
        plt.plot([p[0] for p in counter], [p[1] for p in counter])
        cmap = plt.get_cmap('viridis')
        classes_in_legend = set()
        for tableware_class, smoothed_trajectory in smoothed_trajectories:
            color = cmap(list(predifined_classes.keys()).index(tableware_class)/(len(predifined_classes)))
            x = [point[1][0] for point in smoothed_trajectory]
            y = [point[1][1] for point in smoothed_trajectory]
            alpha_plot = [max(1 - 0.3 * sum(np.diagonal(point[2])), 0) for point in smoothed_trajectory]
            for i in range(len(x)-1):
                if tableware_class in classes_in_legend or alpha_plot[i] < 0.9:
                    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, alpha=alpha_plot[i])
                else:
                    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, alpha=alpha_plot[i], label=tableware_class)
                    classes_in_legend.add(tableware_class)
        plt.axis('equal')
        plt.xlim([1.5, -2.3])
        plt.ylim([1.5, -1.7])
        plt.legend(loc='lower left', bbox_to_anchor=(0.8,-0.1))
        plt.title('trajectories over whole trial')
        plt.savefig(path_prefix + 'mocap.objecttracking.3dtrajetories.png')
            
        # plot all object positions near end of trial in png file to quickly visualize results
        plt.clf()
        table = groundtruth_w['table'].numpy().tolist()
        table.append(table[0])
        plt.plot([p[0] for p in table], [p[1] for p in table])
        counter = groundtruth_w['counter'].numpy().tolist()
        counter.append(counter[0])
        plt.plot([p[0] for p in counter], [p[1] for p in counter])
        for tableware_class, smoothed_trajectory in smoothed_trajectories:
            color = cmap(list(predifined_classes.keys()).index(tableware_class)/(len(predifined_classes)))
            for point in smoothed_trajectory:
                if point[0] == num_frames-100:
                    alpha_plot = max(1 - 0.3 * sum(np.diagonal(point[2])), 0)
                    plt.scatter(point[1][0], point[1][1], color=color, s=5)
                    plt.annotate(tableware_class, (point[1][0], point[1][1]), color=color, fontsize=7, alpha=alpha_plot)
        plt.axis('equal')
        plt.xlim([1.5, -2.3])
        plt.ylim([1.5, -1.7])
        plt.title('object positions near end of trial')
        plt.savefig(path_prefix + 'mocap.objecttracking.finalpositions.png')
        plt.clf()
            
        # write trajectories to lakefs
        with open(path_prefix + 'mocap.objecttracking.3dtrajetories.yaml', 'w') as outfile:
            yaml.dump(smoothed_trajectories, outfile, default_flow_style=False, sort_keys=False)
        tf.keras.backend.clear_session() # memory leak?

if __name__ == '__main__':
    for session in sessions:
        process(session)