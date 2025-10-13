import torch
from torch.utils import data
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision
import aif_model.config as c
import cv2 as cv

class ImageDataset(data.Dataset):
    '''
    Image dataset class
    '''

    def __init__(self, images_root, centroids_path, size = 200000):
        self.imgPaths = list(Path(images_root).rglob('img*.jpg'))
        self.centroids = np.genfromtxt(centroids_path, delimiter=',')
        length = min(size,len(self.imgPaths))

        self.centroids = self.centroids[:length,:]
        self.imgPaths = self.imgPaths[:length]

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, i):
        sample = None
        with Image.open(self.imgPaths[i]) as img:
            try:
                # img = img.resize((c.width,c.height))
                t = torchvision.transforms.functional.pil_to_tensor(img)
                tMin = 0
                tMax = 255
                t = (t - tMin) / (tMax) # Scaling to [0, 1]
                sample=t
            except OSError:
                return self.__getitem__(i-1) # return previous image
            
        lat_rep = np.zeros((c.latent_size))
        # force latent representation
        lat_rep[0] = self.centroids[i][0]
        lat_rep[1] = self.centroids[i][1]
        lat_rep[2] = self.centroids[i][2]

        return sample, lat_rep
    
class IntentionDataset(data.Dataset):
    '''
    Intention dataset class
    '''

    def __init__(self, data_path):
        datas = np.loadtxt(data_path, delimiter=',')
        X = datas[:, :5]  # features
        y = datas[:, 5:]  # labels

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def split_dataset(dataset, percent):
    '''
    Split dataset in train/test set and in batches
    '''

    length = int(dataset.__len__()*percent)
    train_set, test_set = data.random_split(dataset, (length, dataset.__len__() - length))
    train_gen = data.DataLoader(train_set, batch_size=c.n_batch, shuffle=True, num_workers=4)
    test_gen = data.DataLoader(test_set, batch_size=c.n_batch,num_workers=4)

    return train_gen, test_gen

def kl_divergence(p_m, p_v, q_m, log_q_v):
    '''
    Kullback–Leibler divergence
    '''

    return torch.mean(0.5 * torch.sum(torch.log(p_v) - log_q_v + (log_q_v.exp() + (q_m - p_m) ** 2) / p_v - 1, dim=1), dim=0)

def shift_rows(matrix, n):
    '''
    Shifts rows down by n rows
    '''

    shifted_matrix = np.concatenate((matrix[-n:], matrix[:-n]), axis=0)
    return shifted_matrix

def pixels_to_angles(coordinates):
    """
    Translates pixel coordinates into angles in radians
    """

    f = c.width / (2 * np.tan(c.horizontal_fov/2))
    
    cent = (c.width/2, c.height/2) # get center point

    # calculation
    u = -(coordinates[:,0] - cent[0]) # negative because of axis of rotation
    v = coordinates[:,1] - cent[1]

    yaw = np.rad2deg(np.arctan2(u, f))
    pitch = np.rad2deg(np.arctan2(v, f))

    return np.vstack((pitch, yaw)).T # first pitch then yaw

def normalize(x):
    '''
    Normalize angles
    '''

    return x / c.width * 2 - 1

def denormalize(x):
    '''
    Denormalize angles
    '''

    return (x + 1) / 2 * c.width

def add_gaussian_noise(array):
    '''
    Adds gaussian noise to given array
    '''

    sigma = c.noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))

def display_vectors(img, vectors):
    '''
    Displays vectors on image
    '''

    focused=img
    try:
        h,w,_ = img.shape

        red = (w//2 + int(vectors[0,0]*w/2),h//2+int(vectors[0,1]*h/2))
        blue = (w//2 + int(vectors[1,0]*w/2),h//2+int(vectors[1,1]*h/2))
        focus = (w//2 + int(vectors[2,0]*w/2),h//2+int(vectors[2,1]*h/2))

        arrowed = cv.arrowedLine(img.copy(), (w//2,h//2),red,(1,0,0),2)
        # arrowed = cv.arrowedLine(arrowed, (w//2,h//2),blue,(0,0,1),2)

        focused = cv.circle(arrowed.copy(), focus, 5, (0,0.83,0), -1)
    except:
        print("Auto Trials: Vector display fail.")

    return focused

def show_SP(S, P, vectors):
    '''
    Show visual sensory input and visual prediction with vectors indicating sphere belief
    '''

    f = 15
    tmp_S = np.transpose(S[2].detach().squeeze().numpy(),(1,2,0))
    tmp_S = cv.resize(tmp_S,(0,0),fx=f,fy=f)
    tmp_P = np.transpose(P[2],(1,2,0))
    tmp_P = cv.resize(tmp_P,(0,0),fx=f,fy=f)
    tmp_P = display_vectors(tmp_P,vectors)
    combined = np.concatenate((tmp_S,tmp_P), axis=1)
    combined = cv.cvtColor(combined,cv.COLOR_RGB2BGR)
    cv.imshow("S,P",combined)
    cv.waitKey(1)

def gaussian_2d(n, center_x, center_y, sigma):
    '''
    Generate 2D gaussian function for range -1 to 1
    '''

    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, y)

    gaussian_matrix = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)) #1/(sigma * np.sqrt(2*np.pi)) *

    x_deriv = gaussian_matrix * (x - center_x)/sigma**2
    y_deriv = gaussian_matrix * (y - center_y)/sigma**2

    return gaussian_matrix, x_deriv, y_deriv

def log_2d(n, center_x,center_y, amplitude):
    '''
    Generate 2D logarithmic function for range -1 to 1
    '''

    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, y)
    b = 2.6
    d = 1
    
    log_matrix = amplitude * (np.log(-((x - center_x)**2 + (y - center_y)**2) / (b**2) + 1) + d)

    x_deriv = -2*(x-center_x)/(amplitude * b**2 * (1-((x-center_x)**2 + (y-center_y)**2)/b**2))
    y_deriv = -2*(y-center_y)/(amplitude * b**2 * (1-((x-center_x)**2 + (y-center_y)**2)/b**2))
    amp_deriv = (np.log(-((x - center_x)**2 + (y - center_y)**2) / (b**2) + 1) + d)

    return log_matrix, x_deriv, y_deriv, amp_deriv

def pi_foveate(original, mu):
    '''
    Generate visual precision foveated around covert attention center
    '''

    amplitude_idx = c.needs_len+c.prop_len+c.latent_size
    center_x_idx = amplitude_idx + 1
    center_y_idx = amplitude_idx + 2
    log_matrix, x_deriv, y_deriv, amp_deriv = log_2d(c.width, mu[center_x_idx],mu[center_y_idx], mu[amplitude_idx]) # previously gaussian_2d
    pi =  log_matrix * original

    derivative = np.zeros((c.width,c.height,c.needs_len+c.prop_len+c.latent_size+c.focus_len))
    derivative[:,:,amplitude_idx] = amp_deriv * original
    derivative[:,:,center_x_idx] = x_deriv * original
    derivative[:,:,center_y_idx] = y_deriv * original

    dPi_dmu1 = np.zeros((c.height,c.width,c.needs_len+c.prop_len+c.latent_size+c.focus_len))

    return  pi, derivative, dPi_dmu1

def pi_uniform(original, mu):
    '''
    Uniform visual precision
    '''

    return original, np.zeros((c.height,c.width,c.needs_len+c.prop_len+c.latent_size)), np.zeros((c.height,c.width,c.needs_len+c.prop_len+c.latent_size+c.focus_len))

def find_red_centroid(image):
    '''
    Find centroid of biggest red object in the image
    '''

    # Convert to HSV color space
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    # Define red color range (red has two ranges in HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges and combine them
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)

    cx = cy = 0
    
    # Find contours
    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest red region
        largest_contour = max(contours, key=cv.contourArea)
        
        # Compute the centroid
        M = cv.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
    
    return cx, cy


def pi_presence(original, img):
    '''
    Foveate around biggest red object
    '''
    img = np.transpose(img.detach().squeeze().numpy(),(1,2,0))
    img = img * 256 # scale
    r_x, r_y = find_red_centroid(img)

    log_matrix, x_deriv, y_deriv, _ = log_2d(c.width, r_x, r_y, 1.0)
    pi = log_matrix * original
    
    derivative = np.zeros((c.width,c.height,c.needs_len+c.prop_len+c.latent_size+c.focus_len))
    derivative[:,:,-2] = x_deriv * original
    derivative[:,:,-1] = y_deriv * original

    dPi_dS1 = np.zeros((c.height,c.width,c.needs_len+c.prop_len+c.latent_size+c.focus_len))
    return pi, derivative, dPi_dS1