from config import sessions

def process_session(session):
    print('processing session', session)
    from functools import lru_cache
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Stop spam from tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # dont use gpu because it leads to jit compilation error

    from config import xdof, alpha, corners, groundtruth_w, all_camera_names as all_camera_names_gt, trials
    import config
    from d3bv import camera_model, c_in_W

    import tensorflow as tf
    import numpy as np
    import math
    import yaml

    # initial guess of camera pose in world: x,y,z, a_pan, a_tilt, a_roll
    xInitialGuess = tf.constant([3, -1, 2, -np.pi/2, -np.pi/7, 0, # table-side
                                1, -1, 3, np.pi/2, -np.pi/2, 0, # table-top
                                0, 2.5, 3, np.pi, -np.pi/4, 0, # back
                                -1.5, 0.5, 3, np.pi/2, -np.pi/2, 0, # couter-top
                                0, 0, 5, 0, -np.pi/2, np.pi/2, # ceiling
                                0.5, -2.5, 3, -np.pi/12, -np.pi/4, 0, # front
                                0, 0, # table position offset: differance to expected value
                                0, 0], dtype=tf.float64) # counter position offset: differance to expected value

    # get position and orientation for one camera from state x
    def get_params_for_cam(camera_name, x=xInitialGuess):
        idx = all_camera_names_gt.index(camera_name)
        return x[idx*xdof:idx*xdof+xdof]

    def get_params_for_table(table_name, x=xInitialGuess):
        if table_name=='table':
            return tf.add(groundtruth_w['table'], tf.concat([x[-4:-2], [0]], 0))
        elif table_name=='counter':
            return tf.add(groundtruth_w['counter'], tf.concat([x[-2:], [0]], 0))
        elif table_name=='origin':
            return groundtruth_w['origin']

    @lru_cache(maxsize=None) 
    def get_pix_pos(session, trial):
        path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
        with open(f'{path_prefix}mocap.objecttracking.imageannotations.yaml') as f:
            return yaml.safe_load(f)

    # these values are just so tensorflow can compile @tf.function
    found_annotation = False
    for trial in trials:
        try:
            pix_pos_by_cam = get_pix_pos(session, trial)
            all_camera_names = list(pix_pos_by_cam.keys())
            found_annotation = True
            break
        except:
            print('could not get annotation')
            pass
    if not found_annotation:
        return

    predictions = {}

    @tf.function
    def residuals(x, camera_name):
        '''get the error between annotated pixel position and what we expect when we calculate pixel positions from our camera params x and 3d world coordinates'''
        cam_pose = get_params_for_cam(camera_name, x)
        pix_pos_gt = list(pix_pos_by_cam[camera_name]['table'].values())
        pix_pos_gt.extend(list(pix_pos_by_cam[camera_name]['counter'].values()))
        pix_pos_gt.extend(list(pix_pos_by_cam[camera_name]['origin'].values()))
        world_pos_gt = tf.concat([get_params_for_table('table', x=x), get_params_for_table('counter', x=x), get_params_for_table('origin', x=x)], 0)
        world_pos_gt = tf.gather(world_pos_gt, [i for i, pos in enumerate(pix_pos_gt) if pos!=None]) # select cornes we have annottated
        assert world_pos_gt.shape[0] >= 4, f'need at least four courners with annottated pixel positions but got {world_pos_gt.shape[0]} for camera {camera_name}'
        preds = camera_model(world_pos_gt, cam_pose, alpha, (720, 1280, 3), k1=0.0903886121, k2=-0.189294119) # TODO: hardcode k1, k2 in the future because we dont need to recalibrate it with every trial
        pix_pos_gt = tf.constant([coords for coords in pix_pos_gt if not (coords is None)], dtype=tf.float64)
        out = tf.math.reduce_euclidean_norm([preds - pix_pos_gt], axis=2)[0]
        return out

    def residuals_all_cameras(x):
        return tf.concat([residuals(x, camera_name) for camera_name in all_camera_names], axis=0)

    def calc_rmse(errors): # mean root mean squared errors of 4 corners per camera
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(errors))).numpy().tolist()

    # print(f'rmse of initial guess is {calc_rmse(residuals_all_cameras(xInitialGuess))} pixels')

    def gauss_newton_camera(f, xInitialGuess, maxIterations=100, tolerance = 1E-7):
        "Fit's the parameters to the data with Gauss-Newton starting from xInitialGuess"
        x = tf.Variable(xInitialGuess, dtype=tf.float64)
        ctr = 0
        deltaNorm = math.inf
        while ctr<maxIterations and deltaNorm>tolerance:
            res, jac = evalAndJacobian (f, x)
            loss = tf.reduce_sum(res**2)
            delta = tf.linalg.lstsq (jac, res[:,tf.newaxis], l2_regularizer=1E-6)[:,0]
            deltaNorm=tf.norm(delta)
            loss2 = tf.reduce_sum(residuals_all_cameras(x-delta)**2)
            while (math.isnan(loss2) or loss2>=loss) and deltaNorm>tolerance:
                # print(f"{ctr:2} FAILED  loss before {loss:.4f}, delta = {deltaNorm:.5f}, loss after {loss2:.4f}")
                delta = delta / 2
                deltaNorm=tf.norm(delta)
                loss2 = tf.reduce_sum(residuals_all_cameras(x-delta)**2)
            if loss2<loss: x.assign(x - delta) # assign_sub does not work with tensorflow-metal (macOS)
            # if ctr % 10 == 0:
            #     print(f'{ctr:2} SUCCESS loss before {loss:.4f}, delta = {deltaNorm:.5f}, loss after {loss2:.4f}')
            ctr += 1
        return x.value()

    @tf.function
    def evalAndJacobian (f, x):
        '''Returns f(x) and it's Jacobian evaluated as a tensorflow function.'''
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = f(x)
        out = (y, tape.jacobian(y, x))
        return out

    xHat = None
    for trial in trials:
        print(f'processing session {session} trial {trial}')
        try:
            all_camera_names = list(get_pix_pos(session, trial).keys())
        except:
            print('could not download image annottation, skipping trial')
            continue
        pix_pos_by_cam = get_pix_pos(session, trial)
        #this function has to be re-declared for every trial. dont ask why
        @tf.autograph.experimental.do_not_convert
        def residuals_all_cameras(x):
            return tf.concat([residuals(x, camera_name) for camera_name in all_camera_names], axis=0)
        xHat = gauss_newton_camera(residuals_all_cameras, xInitialGuess)
        rmse = calc_rmse(residuals_all_cameras(xHat))
        # print(f'rmse of optimized xHat is {rmse} pixels')
        if rmse < 10:
            xHat_dict = {}
            for camera_name in all_camera_names:
                params = get_params_for_cam(camera_name, x=xHat).numpy().tolist()
                #x,y,z, a_pan, a_tilt, a_roll
                xHat_dict[camera_name] = {'x': params[0], 'y': params[1], 'z': params[2], 'a_pan': params[3], 'a_tilt': params[4], 'a_roll': params[5]}
            xHat_dict['table-offsets-to-correct-position'] = {'table_x': xHat[-4].numpy().tolist(), 'table_y': xHat[-3].numpy().tolist(), 'counter_x': xHat[-2].numpy().tolist(), 'counter_y': xHat[-1].numpy().tolist()}
            path_prefix = f'data/{session}/{trial}/s{str(session).zfill(3)}t{str(trial).zfill(2)}.'
            upload_path = f'{path_prefix}mocap.objecttracking.cameraposes.yaml'
            with open(upload_path, 'w') as outfile:
                yaml.dump(xHat_dict, outfile, default_flow_style=False, sort_keys=False)
                print('wrote', upload_path)
        else:
            print(f'rmse of {rmse} is too high for accurate tracing, maybe annotations are too inaccurate?')

if __name__ == '__main__':
    for session in sessions:
        process_session(session)