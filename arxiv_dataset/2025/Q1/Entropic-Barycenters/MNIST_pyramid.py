import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
from pathlib import Path
import pandas as pd
from scipy import stats, linalg
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fsolve, least_squares, minimize, curve_fit
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
import cvxpy as cp
from sympy import symbols, Eq, solve
from os.path import join
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import h5py
from PIL import Image, ImageDraw, ImageOps
import imageio
import tensorflow as tf
import scipy as sp
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


current_dir = os.getcwd()
sys.path.append(os.path.dirname(current_dir))
from EOT_tools import *
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
sorted_digits = {}
for i in range(10):
    sorted_digits[i] = []
for i in range(len(train_images)):
    label = train_labels[i]
    image = train_images[i]
    image = image / np.sum(image)
    sorted_digits[label] += [image]
ref_images = train_images

def generate_simplex_points(n):
    """
    Generates evenly spaced points on a 2-simplex using n divisions.
    
    :param n: Number of divisions per side.
    :return: List of points on the 2-simplex.
    """
    points = []
    
    # Loop over possible combinations of weights for [x1, x2, x3]
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            points.append([i / n, j / n, k / n])
    
    return np.array(points)

def simplex_to_2d(points):
    """
    Convert 3D simplex points to 2D coordinates for plotting on a triangle.
    
    :param points: Array of points on the 2-simplex.
    :return: Corresponding 2D coordinates for plotting.
    """
    xs = points[:, 0] + 0.5 * points[:, 1]  # x-coordinates in 2D
    ys = np.sqrt(3) / 2 * points[:, 1]      # y-coordinates in 2D
    return xs, ys

def plot_images_on_simplex(points, images,name):
    """
    Plot 28x28 images over the simplex points on a triangle plot without axes, title, or ticks.
    
    :param points: Array of 2-simplex points.
    :param images: List of images (28x28 numpy arrays).
    """
    # Convert simplex points to 2D coordinates
    xs, ys = simplex_to_2d(points)
    
    # Plot the boundary of the triangle (simplex)
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], 'k-')  # triangle edges
    
    # For each point, plot the corresponding image using AnnotationBbox for better handling
    for i, (x, y) in enumerate(zip(xs, ys)):
        image = images[i]
        # Create a small zoomed image to place on the plot
        imagebox = OffsetImage(image, zoom=1.5, cmap='viridis')
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=2)
        plt.gca().add_artist(ab)
    
    # Set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Remove axes, ticks, and title
    plt.axis('off')
    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks
    plt.savefig('gif_folder/analysis_images/{}'.format(name), bbox_inches='tight', pad_inches=0)
    plt.clf()


def plot_points_on_simplex(points1, points2, name):
    """
    Plot two sets of points on the 2-simplex (triangle).
    
    :param points1: First array of 2-simplex points (plotted in red).
    :param points2: Second array of 2-simplex points (plotted in blue).
    :param name: Name of the file to save the plot.
    """
    # Ensure points1 and points2 are NumPy arrays
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Convert both sets of simplex points to 2D coordinates
    xs1, ys1 = simplex_to_2d(points1)
    xs2, ys2 = simplex_to_2d(points2)
    
    # Plot the boundary of the triangle (simplex)
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], 'k-')  # triangle edges
    
    # Plot first set of points in red
    plt.scatter(xs1, ys1, color='red', alpha=0.5, label='Original', zorder=3)
    
    # Plot second set of points in blue
    plt.scatter(xs2, ys2, color='blue', alpha=0.5, label='Recovered', zorder=3)
    
    # Set aspect ratio and remove axes
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    # Show the plot with a legend
    plt.legend()
    plt.savefig(f'gif_folder/synth_images/{name}', bbox_inches='tight', pad_inches=0)
    plt.clf()

def generate_points_and_vector(N):
    # Generate N points uniformly sampled in the square [0,1]^2
    points = np.random.rand(N, 2)
    
    # Create a vector of length N with entries 1/N
    vector = np.full(N, 1/N)
    
    return points, vector



def empirical_to_image(support:np.array, mass:np.array, height=28, width=28, 
                       resolution=5, lower_bound=.0002, bw_method=0.1):
    '''
    empirical_to_image - Converts an empirical measure into an image

    :param support:     (l x 2) np arrayo fo the support loacations
    :param mass:        (l) np array of the mass at each location
    :param height:      positive integer, desired output height
    :param width:       positive integer, desired output width
    :param resolution:  positive integer, upscaling to use in the KDE
    :param lower_bound: float, pixels lower_bound*np.max are set to 0
    :bw_method:         bw_method parameter used in scipy.stats.gaussian_kde
    :return:            (height x width) np array containing the image
    '''

    kde = sp.stats.gaussian_kde(support.T, bw_method=bw_method, weights=mass)

    # creates a 2D grid of locations to evaluate the KDE at
    grid = np.array(
        np.meshgrid(
            np.linspace(0, 1, height * resolution),
            np.linspace(0, 1, width * resolution)
        )
    )
    mesh = grid.reshape(2, height * resolution * width * resolution)

    density = kde(mesh).reshape(height * resolution, width *resolution).T
    #print(np.max(density))
    for i in np.arange(np.shape(density)[0]):
        for j in np.arange(np.shape(density)[1]):
            if density[i][j]<lower_bound*np.max(density):
                density[i][j]=0
    density = density / density.sum()
    blur = np.zeros((height,width))
    for i in range(resolution):
        for j in range(resolution):
            blur += density[i::resolution, j::resolution]
    return blur/blur.sum()

def image_to_empirical(image:np.array):
    '''
    image_to_empirical - Converts an image into an empirical measure which tracks support and mass
                         (the mass is not normalized to a total of 1, but the coordinates are in [0,1]^2)
    
    :param image: (n x m) np array representing an image
    :return:      (l x 2) np array of the support location and (l) np array of mass
    '''

    [height, width] = image.shape

    # for normalizing the height to be between 0 and 1
    # handles the edge case of height or width being 1 pixel
    nheight = max(height - 1, 1) 
    nwidth = max(width - 1, 1)

    support = []
    mass = []
    for i in range(height):
        for j in range(width):
            if image[i,j] == 0:
                continue
                
            support += [[i /  nheight, j / nwidth]]
            mass += [image[i,j]]
            
    return np.array(support), np.array(mass)

def coupling_to_map(coupling, target_support):
    '''
    coupling_to_map - Given a coupling and the target support returns the entropic map
                      evaluated at the source points
    
    :param coupling:       (m x n) np array correspond to a coupling
    :param target_support: (n x 2) np array corresponding to the support of the target
    :return:               (m x 2) np array with the i'th row being the image of the i'th point in the source
    '''
    normalized_rows = coupling / (coupling.sum(1)[:,None])
    return normalized_rows @ target_support

class Trial:
    def __init__(self, digits, indices,score,img_list,eigenvec_list,error_list,mat_list):
        self.digits = digits  # List of digits
        self.indices= indices
        self.score = score    # Score (could be any numeric value)
        self.img_list=img_list
        self.eigenvec_list=eigenvec_list
        self.error_list=error_list
        self.mat_list= mat_list
    
    def __repr__(self):
        return f"Trial(digits={self.digits}, indices={self.indices},score={self.score},)"


#Performs entropy-regularized synthesis and analysis for a measure with image references 
#ref_images is an array of reference images
#int_support is array of 2D points of length n
#int_masses is array of nonnegative numbers of length n adding to 1
#iterations is maximum number of iterations
#tolerance is a stopping criteria, if displacement at k'th iteration is less than tolerance the program will stop
#reg is epsilon
def image_synthesis_and_analysis(ref_images,weights,int_support,int_mass,iterations,tolerance,stepsize,reg):
    m=len(ref_images)
    ref_supports=[]
    ref_masses=[]
    for ref_image in ref_images:
        ref_support, ref_mass = image_to_empirical(ref_image)
        ref_supports.append(ref_support)
        ref_masses.append(ref_mass)
    initial_support=int_support
    initial_mass=int_mass
    for i in np.arange(iterations):
        weighted_maps=[]
        for j in np.arange(m):
            dist = ot.utils.dist(initial_support, ref_supports[j], metric='sqeuclidean') / 2
            if reg == 0:
                coupling = ot.lp.emd(initial_mass, ref_masses[j], dist)
            else:
                coupling = ot.sinkhorn(initial_mass, ref_masses[j], dist, reg)
            weighted_maps += [np.dot(weights[j],coupling_to_map(coupling, ref_supports[j]))]

        map=np.sum(weighted_maps,axis=0)
        displacement=map-initial_support
        squared_displacement_size=np.sum(displacement**2,axis=1)
        gradnorm=np.sum(initial_mass*squared_displacement_size)
        gradnorm=np.linalg.norm(map-initial_support)**.5
        analysis_solution=[]
        if gradnorm<tolerance:
            print('fixed point reached. {} iterations'.format(i))
            weighted_maps=[]
            map_list=[]
            for j in np.arange(m):
                dist = ot.utils.dist(initial_support, ref_supports[j], metric='sqeuclidean') / 2
                if reg == 0:
                    coupling = ot.lp.emd(initial_mass, ref_masses[j], dist)
                else:
                    coupling = ot.sinkhorn(initial_mass, ref_masses[j], dist, reg)
                j_map=coupling_to_map(coupling, ref_supports[j])
                map_list.append(j_map)
                weighted_maps += [np.dot(weights[j],j_map)]
            map=np.sum(weighted_maps,axis=0)
            displacement=map-initial_support
            squared_displacement_size=np.sum(displacement**2,axis=1)
            gradnorm=np.sum(initial_mass*squared_displacement_size)
            gradnorm=np.linalg.norm(map-initial_support)**.5
            print('gradnorm:',gradnorm)
            initial_support=np.dot((1-stepsize),initial_support)+np.dot(stepsize,map)
            #Analysis

            l2norms=[]
            for p in np.arange(m):
                for q in np.arange(m):
                    T_p=map_list[p]-initial_support
                    T_q=map_list[q]-initial_support
                    dotvec=[]
                    for k in np.arange(len(T_p)):
                        innerprod=np.dot(T_p[k],T_q[k])
                        dotvec.append(innerprod)
                    l2diff=np.dot(dotvec,initial_mass)
                    l2norms.append(l2diff)
            l2matrix=np.reshape(l2norms,(m,m))
            x=cp.Variable(m)
            objective=cp.Minimize(cp.quad_form(x,l2matrix))
            constraints=[x>=0,cp.sum(x)==1]
            problem=cp.Problem(objective,constraints)
            problem.solve()
            optimal_x=x.value

            analysis_solution=optimal_x
            reconstruction_error=np.linalg.norm(analysis_solution-weights)
            print('Recovered coefficients:{}'.format(analysis_solution))
            print('Recovery error:{}'.format(reconstruction_error))
            break 
    return initial_support,initial_mass,np.array(analysis_solution),np.array(reconstruction_error),l2matrix


#### IMAGE AND GIF PROCESSING

#Loads empirical measures and analysis vector into memory from saved directories
def load_data(base_dir, i, j):
    """
    Load data for a specific pair of (i, j).

    :param base_dir: Base directory where the data is stored.
    :param i: The i index.
    :param j: The j index.
    :return: Two arrays containing the loaded synth and analysis data.
    """
    synth_path = os.path.join(base_dir, 'synth', str(i), f'iter_{j}.npy')
    analysis_path = os.path.join(base_dir, 'analysis', str(i), f'iter_{j}.npy')
    
    synth_data = None
    analysis_data = None
    
    if os.path.exists(synth_path):
        synth_data = np.load(synth_path)
    else:
        print(f"Warning: {synth_path} does not exist")
    
    if os.path.exists(analysis_path):
        analysis_data = np.load(analysis_path)
    else:
        print(f"Warning: {analysis_path} does not exist")

    return synth_data, analysis_data

#Creates synthesis and analysis pyramid image folders and populates them
def produce_images(iters):
    os.mkdir('gif_folder')
    os.mkdir('gif_folder/synth_images')
    os.mkdir('gif_folder/analysis_images')
    for j in range(iters):
        print(j)
        j_synth_images=[]
        j_analysis_vecs=[]
        for i in i_range:
            synth_data,analysis_data=load_data(base_dir,i,j)
            mass=np.dot(np.ones(len(synth_data)),1/len(synth_data))
            img=empirical_to_image(synth_data,mass)
            j_synth_images.append(img)
            j_analysis_vecs.append(analysis_data)
        plot_images_on_simplex(simplex_points,j_synth_images,'{}'.format(j))
        plot_points_on_simplex(simplex_points,j_analysis_vecs,'{}'.format(j))

#Creates gif from the set of images with title output_gif
def create_pyramid_gif(dir1, dir2, output_gif='output.gif',image_coords=None, duration=0.1, n=None, border_size=10, colormap='viridis', box_size=(100, 100)):
    """
    Create a GIF that plays two image sequences side by side and save the last frame as a PNG.

    :param dir1: Directory containing the first sequence of images.
    :param dir2: Directory containing the second sequence of images.
    :param output_gif: Path to the output GIF file.
    :param duration: Duration of each frame in the GIF (in seconds).
    :param n: Number of images to use from each directory. If None, use all images.
    :param border_size: Size of the white border to add around the images.
    :param colormap: Colormap to apply to the images.
    :param box_size: Size of the boxes for the images.
    """
    def numeric_sort_key(filepath):
        filename = os.path.basename(filepath)
        return int(os.path.splitext(filename)[0])

    images1 = sorted([os.path.join(dir1, img) for img in os.listdir(dir1) if img.endswith('.png')], key=numeric_sort_key)
    images2 = sorted([os.path.join(dir2, img) for img in os.listdir(dir2) if img.endswith('.png')], key=numeric_sort_key)

    if n is not None:
        images1 = images1[:n]
        images2 = images2[:n]

    frames = []
    for img1_path, img2_path in zip(images1, images2):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Ensure both images have the same height
        if img1.size[1] != img2.size[1]:
            new_height = min(img1.size[1], img2.size[1])
            img1 = img1.resize((int(img1.size[0] * new_height / img1.size[1]), new_height))
            img2 = img2.resize((int(img2.size[0] * new_height / img2.size[1]), new_height))

        # Add a white border around each image
        img1 = ImageOps.expand(img1, border=border_size, fill='white')
        img2 = ImageOps.expand(img2, border=border_size, fill='white')

        # Create a new image by concatenating img1 and img2 side by side
        new_img = Image.new('RGB', (img1.width + img2.width, img1.height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))

        # Paste specified images at specified coordinates
        if image_coords:
            for (x, y, overlay_img) in image_coords:
                overlay_img = overlay_img.resize(box_size)
                new_img.paste(overlay_img, (x, y))



            frames.append(new_img)

    # Convert duration from seconds to milliseconds
    duration_ms = int(duration * 1000)

    # Save the frames as a GIF
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)

    # Save the last frame as a PNG
    if frames:
        gif_length = len(frames)
        last_frame = frames[-1]
        last_frame.save(f'frame_{gif_length}.png')

def apply_colormap(image, colormap):
    normed_data = (image - np.min(image)) / (np.max(image) - np.min(image))
    colormap = cm.get_cmap(colormap)
    mapped_data = colormap(normed_data)
    return (mapped_data[:, :, :3] * 255).astype(np.uint8)

####SCRIPT

trial_list_unif = []
simplex_points = generate_simplex_points(5)

points1,mass1=image_to_empirical(ref_images[21])
points2,mass2=image_to_empirical(ref_images[12])
points3,mass3=image_to_empirical(ref_images[9])

three_image=ref_images[21]
zero_image=ref_images[12]
four_image=ref_images[9]
three_image_colored = apply_colormap(three_image, 'viridis')
zero_image_colored = apply_colormap(zero_image, 'viridis')
four_image_colored = apply_colormap(four_image, 'viridis')
three_image_pil = Image.fromarray(three_image_colored)
zero_image_pil = Image.fromarray(zero_image_colored)
four_image_pil = Image.fromarray(four_image_colored)


ref_indices = [9, 12, 21]
selected_ref_images=[]
for j in ref_indices:
    # ref_index = random.randrange(len(sorted_digits[j]))
    ref_image = ref_images[j]
    a_i, b_i = image_to_empirical(ref_image)
    ref_image = empirical_to_image(a_i, b_i)
    selected_ref_images.append(ref_image)


# Produces synthesis and analysis data run for #iterations and saves them in directories

iterations = np.arange(1000)
base_dir = ''
i_range = range(21) 
j_range=iterations
n_divisions = 5
'''
for lam in simplex_points:
    try:
        index_of_lam = np.where(np.all(simplex_points == lam, axis=1))[0][0]
        lam_str = str(index_of_lam)  # Convert lam to a string for directory names
        os.makedirs(f'synth/{lam_str}', exist_ok=True)
        os.makedirs(f'analysis/{lam_str}', exist_ok=True)
        os.makedirs(f'error/{lam_str}', exist_ok=True)
        # Synthesize a BC for each lambda
        int_support, int_mass = generate_points_and_vector(2000)
        for k in iterations:
            np.save(f'synth/{lam_str}/iter_{k}', int_support)
            int_support, int_mass, analysis_solution_unif, reconstruction_error_unif, l2matrix_unif = image_synthesis_and_analysis(
                np.array(selected_ref_images), lam,int_support,int_mass, 1, 100, 0.025, 0.0002
            )
            np.save(f'analysis/{lam_str}/iter_{k}', analysis_solution_unif)
            np.save(f'error/{lam_str}/iter_{k}', reconstruction_error_unif)

    except Exception as e:
        print(f"Error processing lambda {lam}: {e}")
        # Skip this lambda if there's an error and move to the next one
        continue

'''


#Produces a gif visualizing the evolution of the synthesis and analysis algorithms
#Saves the last frame as an .png
gif_length = len(iterations)  # Number of images to use in gif (default uses all images produced)
#produce_images(gif_length)
output_gif = 'output.gif'
dir1 = 'gif_folder/analysis_images'
dir2 = 'gif_folder/synth_images'
image_coords = [(261, 18, three_image_pil), (25, 450, zero_image_pil), (498, 450, four_image_pil)]

create_pyramid_gif(dir1, dir2, output_gif, image_coords=image_coords, border_size=80, box_size=(55, 55))

