import os
import sys
import math
import random
import datetime
import numpy as np
import imageio as iio
import time
from shutil import copyfile
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from fcnpytorch.torchfcn.models import FCN16s as FCN16s
from fcnpytorch.torchfcn.models import FCN8s as FCN8s
from utils import StatCollector, normalize_data, exclude_months, interpolate_missing, get_random_crop


# Global vars
BASE_PATH_DATA = 'data-extended-smhi'
BASE_PATH_LOG = 'log'
SEED = 0
USE_GPU = True
FCN_TYPE = 'fcn8s'  # 'fcn8s' or 'fcn16s'
CROP_SIZE = 100
MIN_CROP_FRAME = 10  # Require measurement station(s) to be at least a minimum distance from the crop border
NBR_RAIN_INPUT = 20  # use t-x+1, t-x+2, ..., t as measurement inputs, where x=NBR_RAIN_INPUT
NBR_RAIN_INPUT_COARSE = 0  # Coarser time-scale, can provide longer history of rainfalls
RAIN_INPUT_END = 1  # number of days before prediction day that the data stops
NBR_TEMP_INPUT = 20  # Number of past days of temperature measurements to use
NBR_TEMP_INPUT_COARSE = 0  # Coarser time-scale, can provide longer history of temperatures
TEMP_INPUT_END = 1  # number of days before prediction day that the data stops
RELATIVE_HEIGHT_MAP = True  # height map's minimum value is 0 if True
NBR_FLOW_INPUT = 0  # Defaults 0 -- i.e. no past flows are used as input
FLOW_INPUT_END = 0  # number of days before prediction day that the data stops
BATCH_SIZE = 64
LOSS_TYPE = 'huber'  # 'huber' or 'mse'
NUM_TRAIN_BATCHES = 250000
OPTIMIZER = 'adam'
LR = 0.0002
WEIGHT_DECAY = 0
MOMENTUM = 0.9
BETA1 = 0.5  # for ADAM
RAIN_FLOW_TEMP_NORMALIZATION = '01'
DATA_AUGMENTATIONS = ['left-right', 'up-down']  # Can use 'left-right' and 'up-down' data augmentation
MODEL_LOAD_PATH = None
EVAL_ONLY = True  # True --> No backprop is used
LOCATIONS = ['Jonkoping', 'Knislinge', 'Krycklan', 'Skivarp', 'Skovde', 'Torup', 'Tumba',
             'Dalbergsan', 'Degea', 'Hassjaan', 'Lillan', 'Lillan_blekinge']
LOCATIONS_TRAIN = ['Knislinge', 'Krycklan', 'Skivarp', 'Skovde', 'Torup', 'Tumba', 'Dalbergsan', 'Degea', 'Lillan_blekinge']
LOCATIONS_VAL = ['Jonkoping', 'Hassjaan', 'Lillan']
USE_IDXS_KRYCKLAN = [1, 2, 20, 4, 5, 6, 7]
EXCLUDE_MONTHS = [4, 5, 9]
EXCLUDE_MONTHS_ONLY_KRYCKLAN = True
SAVE_MODEL_EVERY_KTH = 50000
SAVE_PREDICTION_VISUALIZATAIONS = True

# Listing of number of days per month
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("training.py", os.path.join(log_dir, "training.py"))
copyfile("fcnpytorch/torchfcn/models/fcn8s.py", os.path.join(log_dir, "fcn8s.py"))

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set max number of threads to use to 10 (dgx1 limit)
torch.set_num_threads(10)

######################### DATA READING START ###################################

# Read rainfall and temperature data
rain_datas = {}
temp_datas = {}
for loc in LOCATIONS:
    rain_data = scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                 'rain_data.mat'))['rain_data'][0][0]
    rain_datas[loc] = rain_data[1]
    temp_data = scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                 'temp_data.mat'))['temp_data'][0][0]
    temp_datas[loc] = temp_data[1]
rain_datas = normalize_data(rain_datas, RAIN_FLOW_TEMP_NORMALIZATION)
temp_datas = normalize_data(temp_datas, RAIN_FLOW_TEMP_NORMALIZATION)
interpolate_missing(rain_datas)
interpolate_missing(temp_datas)

# Read waterflow measurements + dates (same for rain and all flow measurements)
flow_datas = {}
all_dates = {}
max_flow = 0
for loc in LOCATIONS:
    flow_data = scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                 'flow_data.mat'))['flow_data'][0]
    ids = np.squeeze(np.asarray([flow_data[i][0][0][2] for i in range(len(flow_data))]))
    all_dates[loc] = np.squeeze(np.asarray([flow_data[i][0][0][0] for i in range(len(flow_data))]))
    if all_dates[loc].ndim == 3:
        all_dates[loc] = all_dates[loc][0]
    flow_data_coords = np.squeeze(np.asarray([flow_data[i][0][0][3] for i in range(len(flow_data))]))
    if flow_data_coords.ndim == 1:
        flow_data_coords = flow_data_coords[np.newaxis, :]
    flow_data = np.squeeze(np.asarray([flow_data[i][0][0][1] for i in range(len(flow_data))])).T
    if flow_data.ndim == 1:
        flow_data = flow_data[:, np.newaxis]
    if loc == 'Krycklan':
        to_keep = []
        for i in USE_IDXS_KRYCKLAN:
            to_keep.append(np.nonzero(ids == i)[0][0])
        flow_data_coords = flow_data_coords[to_keep, :]
        flow_data = flow_data[:, to_keep]
    flow_datas[loc] = {'coords': flow_data_coords, 'data_in': np.copy(flow_data), 'data_gt': np.copy(flow_data)}
    max_flow = max(max_flow, np.nanmax(flow_data))
flow_datas = normalize_data(flow_datas)
for loc in LOCATIONS:
    flow_datas[loc]['data_gt'] /= max_flow
    if NBR_FLOW_INPUT > 0:
        flow_datas[loc]['data_in'] = interpolate_missing(flow_datas[loc]['data_in'])

# Read images
concat_imgs = {}
maxes = np.zeros(10)  # satellite has 3 channels (RGB)
for loc in LOCATIONS:
    hyd_conn_gray = np.float32(scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                               'hydraulic_conductivity_gray.mat'))['data_gray'])[:, :, np.newaxis]
    soil_type_gray = np.float32(scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                'soil_type_gray.mat'))['data_gray'])[:, :, np.newaxis]
    soil_depth_gray = np.float32(scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                'soil_depth_gray.mat'))['data_gray'])[:, :, np.newaxis]
    soil_moisture_gray = np.float32(scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                    'soil_moisture_gray.mat'))['data_gray'])[:, :, np.newaxis]
    soil_cover_gray = np.float32(scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                 'soil_cover_gray.mat'))['data_gray'])[:, :, np.newaxis]
    height_map_gray = np.float32(scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                 'height_map_gray.mat'))['data_gray'])[:, :, np.newaxis]
    terrain_gray = np.float32(scipy.io.loadmat(os.path.join(BASE_PATH_DATA, loc,
                                 'terrain_gray.mat'))['data_gray'])[:, :, np.newaxis]
    satellite = np.float32(iio.imread(os.path.join(BASE_PATH_DATA, loc, 'satellite.png')))

    # Concatenate into "super image" along the depth dimension
    all_imgs = [hyd_conn_gray, soil_type_gray, soil_depth_gray, soil_moisture_gray,
                soil_cover_gray, height_map_gray, terrain_gray, satellite]
    concat_imgs[loc] = np.concatenate(all_imgs, axis=2)

    # Keep track of things for image normalization
    for i in range(len(maxes)):
        maxes[i] = max(maxes[i], np.max(concat_imgs[loc][:, :, i]))
        
# Unlikely to be needed, but used to avoid divide-with-zeros
maxes[maxes == 0] = 1

# Normalize images
for loc in LOCATIONS:
    concat_imgs[loc] /= maxes

# Extract and set certain dimensionalities
H, W, C = concat_imgs[LOCATIONS[0]].shape
dim_input = C + NBR_RAIN_INPUT + NBR_RAIN_INPUT_COARSE + NBR_TEMP_INPUT \
            + NBR_TEMP_INPUT_COARSE + NBR_FLOW_INPUT

# Setup model
if FCN_TYPE == 'fcn8s':
    model = FCN8s(n_class=1, dim_input=dim_input, weight_init='normal')
elif FCN_TYPE == 'fcn16s':
    model = FCN16s(n_class=1, dim_input=dim_input, weight_init='normal')
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
if not MODEL_LOAD_PATH is None:
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
model.to(device)

# Setup loss
if LOSS_TYPE == 'mse':
    criterion_elwise = nn.MSELoss(reduce=False)
    criterion_other = nn.HuberLoss(reduction='none', delta=1.0)
elif LOSS_TYPE == 'huber':
    criterion_elwise = nn.HuberLoss(reduction='none', delta=1.0)
    criterion_other = nn.MSELoss(reduce=False)

# Setup optimizer.
if OPTIMIZER == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.999))
elif OPTIMIZER == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

# Setup things related to train-val split
start_value = max(NBR_RAIN_INPUT + 10 * NBR_RAIN_INPUT_COARSE + RAIN_INPUT_END,
                  NBR_TEMP_INPUT + 10 * NBR_TEMP_INPUT_COARSE + TEMP_INPUT_END,
                  NBR_FLOW_INPUT + FLOW_INPUT_END)
if False and VAL_OPTS['use_rain_interval']:
    rain_rand_nbrs_train = [i for i in range(start_value, VAL_OPTS['rain_interval'][0])] \
                            + [i for i in range(VAL_OPTS['rain_interval'][1] + start_value, nbr_meas)]
    rain_rand_nbrs_val = [i for i in range(VAL_OPTS['rain_interval'][0], VAL_OPTS['rain_interval'][1])]
else:
    rain_rand_nbrs_train_all = {}
    rain_rand_nbrs_val_all = {}
    for loc in LOCATIONS:
        if not EXCLUDE_MONTHS_ONLY_KRYCKLAN or loc == 'Krycklan':
            rain_rand_nbrs_train, rain_rand_nbrs_val = \
                exclude_months(range(start_value, rain_datas[loc].shape[0]),
                               range(start_value, rain_datas[loc].shape[0]),
                               all_dates[loc], NBR_RAIN_INPUT, EXCLUDE_MONTHS)
        else:
            rain_rand_nbrs_train, rain_rand_nbrs_val = \
                exclude_months(range(start_value, rain_datas[loc].shape[0]),
                               range(start_value, rain_datas[loc].shape[0]),
                               all_dates[loc], NBR_RAIN_INPUT, [])
        rain_rand_nbrs_train_all[loc] = rain_rand_nbrs_train
        rain_rand_nbrs_val_all[loc] = rain_rand_nbrs_val

# Setup StatCollector
sc = StatCollector(stat_train_dir, NUM_TRAIN_BATCHES, 10)
sc.register('L2_loss', {'type': 'avg', 'freq': 'step'})
sc.register('L2_loss_val', {'type': 'avg', 'freq': 'step'})
sc.register('L2_loss-other', {'type': 'avg', 'freq': 'step'})
sc.register('L2_loss-other_val', {'type': 'avg', 'freq': 'step'})
for i in range(12):
    sc.register('L2_loss-month-' + str(i + 1), {'type': 'avg', 'freq': 'step'})
    sc.register('L2_loss-month-' + str(i + 1) + '_val', {'type': 'avg', 'freq': 'step'})

def _forward_compute(rain_rand_nbrs, sc, mode='train'):
    # Format current batch (including ground truth)
    data_batch = np.zeros((BATCH_SIZE, dim_input, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    gt_batch = np.zeros((BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    indicator_batch = np.zeros((BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    meas_mostly_in_month = np.zeros(BATCH_SIZE, dtype=np.int32)
    b = 0
    while b < BATCH_SIZE:

        # Randomly select location at which to predict the water flow
        # Also sample data augmentation during training
        aug_lr_flip = False
        aug_ud_flip = False
        if mode == 'train':
            loc = random.choice(LOCATIONS_TRAIN)

            # Sample data augmentation
            for aug in DATA_AUGMENTATIONS:
                if aug == 'left-right':
                    # Flips image left-right with probability 50%
                    aug_lr_flip = random.choice([False, True])
                if aug == 'up-down':
                    # Flips image up-down with probability 50%
                    aug_ud_flip = random.choice([False, True])
        else:
            loc = random.choice(LOCATIONS_VAL)

        # Randomly select time point at which to predict the water flow
        rand_nbr = random.choice(rain_rand_nbrs[loc])

        # Extract crop and the measurement points that are inside the crop
        crop, inside, idxs_inside = get_random_crop(H, W, CROP_SIZE, flow_datas[loc]['coords'], MIN_CROP_FRAME)
        inside = np.int32(np.round(inside))
        if RELATIVE_HEIGHT_MAP:
            img_crop = np.copy(concat_imgs[loc][crop[0]:crop[1], crop[2]:crop[3], :])
            img_crop[:, :, 5] -= np.min(img_crop[:, :, 5])
        else:
            img_crop = concat_imgs[loc][crop[0]:crop[1], crop[2]:crop[3], :]

        # Extract ground truth flow measurements from the measurement
        # points inside the crop
        flows = flow_datas[loc]['data_gt'][rand_nbr - 1, idxs_inside]
        idxs_nan = np.isnan(flows)
        if np.count_nonzero(~idxs_nan) == 0:
            # Cannot accept batches with no measurements -- try again
            continue
        idxs_inside = idxs_inside[~idxs_nan]
        inside = inside[~idxs_nan, :]
        flows = flows[~idxs_nan]
        flow_map = np.zeros((H, W), dtype=np.float32)
        flow_map[inside[:, 0], inside[:, 1]] = flows
        flow_map = flow_map[crop[0]:crop[1], crop[2]:crop[3]]

        # Extract previous flow measurements to be used as input
        flows_in = flow_datas[loc]['data_in'][rand_nbr - NBR_FLOW_INPUT - FLOW_INPUT_END : rand_nbr - FLOW_INPUT_END, idxs_inside]
        flow_map_in = np.zeros((H, W, NBR_FLOW_INPUT), dtype=np.float32)
        flow_map_in[inside[:, 0], inside[:, 1], :] = flows_in.T
        flow_map_in = flow_map_in[crop[0]:crop[1], crop[2]:crop[3], :]

        # Create indicator of which measurement points fall inside the crop
        indicator_crop = np.zeros((H, W), dtype=bool)
        indicator_crop[inside[:, 0], inside[:, 1]] = 1
        indicator_crop = indicator_crop[crop[0]:crop[1], crop[2]:crop[3]]

        # Extract rainfall time series (of length NBR_RAIN_INPUT)
        rain = rain_datas[loc][rand_nbr - NBR_RAIN_INPUT - RAIN_INPUT_END : rand_nbr - RAIN_INPUT_END]
        rain = np.tile(np.reshape(rain, [1, 1, NBR_RAIN_INPUT]), [CROP_SIZE, CROP_SIZE, 1])

        # Extract coarse rain information from further back in time
        rain_coarse = rain_datas[loc][rand_nbr - NBR_RAIN_INPUT - NBR_RAIN_INPUT_COARSE * 10 : rand_nbr - NBR_RAIN_INPUT]
        rain_coarse = np.array([np.mean(rain_coarse[i : i + 10]) for i in range(NBR_RAIN_INPUT_COARSE)])
        rain_coarse = np.tile(np.reshape(rain_coarse, [1, 1, NBR_RAIN_INPUT_COARSE]), [CROP_SIZE, CROP_SIZE, 1])

        # Extract temperature time series (of length NBR_TEMP_INPUT)
        temp = temp_datas[loc][rand_nbr - NBR_TEMP_INPUT - TEMP_INPUT_END : rand_nbr - TEMP_INPUT_END]
        temp = np.tile(np.reshape(temp, [1, 1, NBR_TEMP_INPUT]), [CROP_SIZE, CROP_SIZE, 1])

        # Extract coarse temperature information from further back in time
        temp_coarse = temp_datas[loc][rand_nbr - NBR_TEMP_INPUT - NBR_TEMP_INPUT_COARSE * 10 : rand_nbr - NBR_TEMP_INPUT]
        temp_coarse = [np.mean(temp_coarse[i : i + 10]) for i in range(NBR_TEMP_INPUT_COARSE)]
        temp_coarse = np.tile(np.reshape(temp_coarse, [1, 1, NBR_TEMP_INPUT_COARSE]), [CROP_SIZE, CROP_SIZE, 1])

        # Extract corresponding dates and convert to a scalar in [-1, 1] which
        # indicates roughly what time of year it is
        dates = all_dates[loc][rand_nbr - start_value : rand_nbr, :]

        frac_meases = np.zeros(12)
        for i in range(12):
            frac_meases[i] = np.count_nonzero(dates[:, 1] == i + 1) / max(1, NBR_RAIN_INPUT)
        meas_mostly_in_month[b] = np.argmax(frac_meases)

        # Perform data augmentation
        curr_data_batch = np.concatenate([img_crop, flow_map_in, rain, rain_coarse, temp, temp_coarse], axis=2)
        curr_data_batch = np.transpose(curr_data_batch, [2, 0, 1])
        if aug_lr_flip:
            curr_data_batch = np.flip(curr_data_batch, axis=2)
            indicator_crop = np.flip(indicator_crop, axis=1)
            flow_map = np.flip(flow_map, axis=1)
        if aug_ud_flip:
            curr_data_batch = np.flip(curr_data_batch, axis=1)
            indicator_crop = np.flip(indicator_crop, axis=0)
            flow_map = np.flip(flow_map, axis=0)

        # Insert batch elements
        data_batch[b, :, :, :] = curr_data_batch
        indicator_batch[b, :, :, :] = indicator_crop[np.newaxis, :, :]
        gt_batch[b, :, :, :] = flow_map[np.newaxis, :, :]

        # Continue constructing the next element of the batch
        b += 1

    # Send to device (typically GPU)
    data_batch = torch.tensor(data_batch).to(device)
    gt_batch = torch.tensor(gt_batch).to(device)
    indicator_batch = torch.tensor(indicator_batch).to(device)

    # Forward the batch through the model, then compute the loss
    flow_pred = model(data_batch)
    err_pred_elwise = criterion_elwise(flow_pred, gt_batch)
    err_pred_sep_batch = torch.sum(err_pred_elwise * indicator_batch,
                                   dim = [1,2,3]) / torch.sum(indicator_batch,
                                                              dim=[1,2,3])
    # Track also the other type of loss (Huber if MSE is loss, and MSE if Huber is loss)
    err_pred_other = criterion_other(flow_pred, gt_batch)
    err_pred_other_sep_batch = torch.sum(err_pred_other * indicator_batch,
                                   dim = [1,2,3]) / torch.sum(indicator_batch,
                                                              dim=[1,2,3])
    err_pred = torch.mean(err_pred_sep_batch)
    err_pred_other = torch.mean(err_pred_other_sep_batch)
    if mode == 'train':
        sc.s('L2_loss').collect(err_pred.item())
        sc.s('L2_loss-other').collect(err_pred_other.item())
        for i in range(12):
            if np.count_nonzero(meas_mostly_in_month == i) > 0 and False:
                err_pred_month_i = torch.mean(err_pred_sep_batch[meas_mostly_in_month == i])
                sc.s('L2_loss-month-' + str(i + 1)).collect(err_pred_month_i.item())
    else:
        sc.s('L2_loss_val').collect(err_pred.item())
        sc.s('L2_loss-other_val').collect(err_pred_other.item())
        for i in range(12):
            if np.count_nonzero(meas_mostly_in_month == i) > 0 and False:
                err_pred_month_i = torch.mean(err_pred_sep_batch[meas_mostly_in_month == i])
                sc.s('L2_loss-month-' + str(i + 1) + '_val').collect(err_pred_month_i.item())

    # Return the error of the prediction, one of the predicted flow maps, the
    # coordinates of the measurement sites, and the associated spatial inputs
    return err_pred, [np.squeeze(flow_pred.cpu().detach().numpy()[0, 0, :, :]),
                      np.squeeze(indicator_batch.cpu().detach().numpy()[0, 0, :, :]),
                      np.transpose(np.squeeze(data_batch.cpu().detach().numpy()[0, :10, :, :]), [1, 2, 0])]

# Main training loop
print("Starting training loop...")
if EVAL_ONLY:
    model.train()
else:
    model.eval()
for it in range(NUM_TRAIN_BATCHES):

    # Forward computation
    err_pred, _ = _forward_compute(rain_rand_nbrs_train_all, sc, mode='train')

    # Calculate gradients in backward pass and update model weights
    if not EVAL_ONLY:
        optimizer.zero_grad()
        err_pred.backward()
        optimizer.step()

    # Occassionally save model weights
    if (not EVAL_ONLY) and (SAVE_MODEL_EVERY_KTH is not None and SAVE_MODEL_EVERY_KTH > 0) and \
        ((it > 0 and (it + 1) % SAVE_MODEL_EVERY_KTH == 0) or it + 1 == NUM_TRAIN_BATCHES):
        torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % (it + 1)))

    # Track training and validation statistics
    if it % 10 == 0:
        _, pred_and_input = _forward_compute(rain_rand_nbrs_val_all, sc, mode='val')
        sc.print()
        #sc.plot()
        sc.save()
        print("Iter: %d / %d" % (it, NUM_TRAIN_BATCHES))

        # Save the predicted flow map and corresponding height map
        if SAVE_PREDICTION_VISUALIZATAIONS and it % 100 == 0:
            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(2,4,1)
            plt.imshow(pred_and_input[0])
            plt.title('Predicted flow')
            plt.axis('off')

            fig.add_subplot(2,4,2)
            plt.imshow((pred_and_input[2][:,:,5] + 1) / 2)
            plt.title('Elevation map')
            plt.axis('off')

            fig.add_subplot(2,4,3)
            plt.imshow((pred_and_input[2][:,:,6] + 1) / 2)
            plt.title('Terrain slope map')
            plt.axis('off')

            fig.add_subplot(2,4,4)
            plt.imshow(pred_and_input[2][:,:,7:])
            plt.title('Satellite')
            plt.axis('off')

            fig.add_subplot(2,4,5)
            plt.imshow((pred_and_input[2][:,:,1] + 1) / 2)
            plt.title('Soil type')
            plt.axis('off')

            fig.add_subplot(2,4,6)
            plt.imshow((pred_and_input[2][:,:,2] + 1) / 2)
            plt.title('Soil depth')
            plt.axis('off')

            fig.add_subplot(2,4,7)
            plt.imshow((pred_and_input[2][:,:,3] + 1) / 2)
            plt.title('Soil moisture')
            plt.axis('off')

            fig.add_subplot(2,4,8)
            plt.imshow((pred_and_input[2][:,:,4] + 1) / 2)
            plt.title('Land cover')
            plt.axis('off')

            plt.savefig(os.path.join(stat_train_dir, 'pred_flow_%d.png' % it))
            plt.cla()
            plt.clf()
            plt.close('all')

print("Training completed!")
