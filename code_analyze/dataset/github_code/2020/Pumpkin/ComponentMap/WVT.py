'''
DEFINE INPUTS IN MAIN FUNCTION AT END OF FILE
This file will create a WVT map given a set of pixels (x and y coordinates) and
a weight. This is specifically written for X-Ray spectroscopy, but could be easily
canabilized for any other application necessitating WVT.
---------------------------------------------------
---------------------------------------------------
Inputs:
    image_fits - Name of Image Fits File (e.g. "simple_image.fits")
    exposure_map - Name of exposure map -- optional -- (e.g. 'flux_broad.expmap')
    stn_target - Target Signal-to-Noise (e.g. 50)
    pixel_size - Pixel radius in degrees (e.g. 0.492 for CHANDRA AXIS I)
    tol - tolerance Level (e.g. 1e-16)
    roundness_crit - Critical Value Describing Roundness of Bin (e.g. 0.3)
    home_dir - Full Path to Home Directory (e.g. '/home/user/Documents/ChandraData/12833/repro')
    image_dir - Full Path to Image Directory (e.g. '/home/user/Desktop')
    output_dir - Full Path to Output Directory (e.g. '/home/user/Documents/ChandraData/12833')
---------------------------------------------------
List of Functions (In order of appearance):
    plot_Bins --> Plot bins with five alternating color schemes
    plot_Bins_SN --> Plot bins with color based of Signal-to-Noise value
    read_in --> Read in flux file and gather flux information
    Nearest_Neighbors --> Calculate nearest neighbors using builtin sklearn algorithm
    dist --> Calculates Euclidean distances
    closest_node --> Calculates closest node to bin centroid
    adjacency --> Determines if pixels are adjacent (diagonals don't count)
    Roundness --> Calculates roundness parameter of bin
    Potential_SN --> Calculates Potential New Signal-to-Noise if pixel is added
    Bin_data --> Creates data for Chandra with only unique pixels from read_in
    bin_num_pixel --> Calculates bixel's associated bin (i.e. in which bin is my pixel??)
    assigned_missing_pixels --> Assign pixels which weren't read in due to 0 Signal-to-Noise
    Bin_Acc --> Perform Bin Accretion Algorithm
    converged_met --> Check that all bins have converged in WVT
    Rebin_Pixels --> Rebin pixels in WVT algorithm
    WVT --> Perform Weighted Voronoi Tessellation Algorithm
---------------------------------------------------
Outputs:
    -- Creates bin plots for bin accretion and WVT algorithms
    -- Creates text files for each bin containing chip information and bin number
                    pixel (x,y) are CENTER coordinates!
---------------------------------------------------
TO DO:

---------------------------------------------------
---------------------------------------------------
Carter Rhea
https://carterrhea.com
carterrhea93@gmail.com
'''

#-----------------INPUTS--------------------------#
import os
import sys
import numpy as np
import statistics as stats
from astropy.io import fits
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
from read_input import read_input_file
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
# Plot Bins
#   parameters:
#       Bin - list of bin objects
#       x_min - minimum x value of pixel
#       x_max - maximum x value of pixel
#       y_min - minimum y value of pixel
#       x_max - maximum y value of pixel
#       stn_target - target signal-to-noise value
#       file_dir - Full Path to location of new image
#       fileame - name of file to be printed
def plot_Bins(Bins,x_min,x_max,y_min,y_max,stn_target,file_dir,filename):
    # Set everything up for the plotting...
    fig = plt.figure()
    ax = plt.axes(xlim=(x_min,x_max), ylim=(y_min,y_max))
    N = len(Bins)
    StN_list = []
    SNR_list = []
    bin_nums = []
    max_StN = max([bin.StN[0] for bin in Bins])
    StN_list = [bin.StN[0] for bin in Bins]
    median_StN = np.median(StN_list)
    stand_dev = stats.stdev(StN_list)
    mini_pallete = ['mediumspringgreen','salmon','cyan','orchid','yellow','blue','red','magenta','black','white']
    binNumber  = 0
    # Step through each bin and assign pixels/create their patch
    for bin in Bins:
        bin_nums.append(bin.bin_number)
        SNR = bin.StN[0]/median_StN
        SNR_list.append(SNR)
        for pixel in bin.pixels:
            x_coord = pixel.pix_x
            y_coord = pixel.pix_y
            if binNumber%10 == 0:
                color = mini_pallete[0]
            if binNumber%10 == 1:
                color = mini_pallete[1]
            if binNumber%10 == 2:
                color = mini_pallete[2]
            if binNumber%10 == 3:
                color = mini_pallete[3]
            if binNumber%10 == 4:
                color = mini_pallete[4]
            if binNumber%10 == 5:
                color = mini_pallete[5]
            if binNumber%10 == 6:
                color = mini_pallete[6]
            if binNumber%10 == 7:
                color = mini_pallete[7]
            if binNumber%10 == 8:
                color = mini_pallete[8]
            if binNumber%10 == 9:
                color = mini_pallete[9]
            #Shift because x_coord,y_coord are the center points
            rectangle = plt.Rectangle((x_coord,y_coord),1,1, fc=color)
            ax.add_patch(rectangle)
        binNumber += 1
    centroids_x = [Bins[i].centroidx[0] for i in range(len(Bins))]
    centroids_y = [Bins[i].centroidy[0] for i in range(len(Bins))]
    # Plot
    ax.scatter(centroids_x,centroids_y,marker='+',c="black")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bin Mosaic")
    plt.savefig(file_dir+'/'+filename+".png")
    plt.clf()
    plt.hist(StN_list)
    plt.xlim((median_StN-3*stand_dev,median_StN+3*stand_dev))
    plt.ylabel("Number of Bins")
    plt.xlabel("Signal-to-Noise")
    plt.title("Signal-to-Noise per Bin")
    plt.savefig(file_dir+'/histograms/'+filename+".png")
    plt.clf()
    SNR_std = stats.stdev(SNR_list)
    SNR_med = np.median(SNR_list)
    SNR_nel = len(SNR_list)
    n_el = SNR_nel #Just change this
    plt.scatter(np.arange(n_el),SNR_list, marker = '+', color='salmon',label="Data Points")
    plt.plot(np.arange(n_el),[SNR_med for i in range(n_el)], linestyle='solid', color= 'forestgreen', label="Median")
    plt.plot(np.arange(n_el),[SNR_med+SNR_std for i in range(n_el)], linestyle='--', color= 'black')
    plt.plot(np.arange(n_el),[SNR_med-SNR_std for i in range(n_el)], linestyle='--', color= 'black', label="sigma")
    plt.title("Signal to Noise Ratio")
    plt.ylim((min(SNR_list),max(SNR_list)))
    plt.xlabel("Bin Number")
    plt.ylabel("Signal-to-Noise Normalized by Median Value")
    plt.legend(loc='center left',bbox_to_anchor=(1.0, 0.5),
          ncol=1, fancybox=True, shadow=True)
    plt.savefig(file_dir+'/'+filename+"_scatter.png", bbox_inches="tight")
    plt.clf()
    return None
#-------------------------------------------------#
#-------------------------------------------------#
# Plot plot_Bins_SN
#   parameters:
#       Bin - list of bin objects
#       x_min - minimum x value of pixel
#       x_max - maximum x value of pixel
#       y_min - minimum y value of pixel
#       x_max - maximum y value of pixel
#       file_dir - Full Path to location of new image
#       fileame - name of file to be printed
def plot_Bins_SN(Bins,x_min,x_max,y_min,y_max,file_dir,filename):
    fig = plt.figure()
    fig.set_size_inches(7, 7)
    ax = plt.axes(xlim=(x_min,x_max), ylim=(y_min,y_max))
    N = len(Bins)
    cmap = mpl.cm.get_cmap('viridis')
    max_StN = max([bin.StN[0] for bin in Bins])
    median_StN = np.median([bin.StN[0] for bin in Bins])
    patches = []
    SN_list = []
    SNR_list = []
    bin_nums = []
    for bin in Bins:
        bin_nums.append(bin.bin_number)
        #Color based on signal-to-noise value
        SNR = bin.StN[0]/median_StN
        SN_list.append(bin.StN[0])
        SNR_list.append(SNR)
        for pixel in bin.pixels:
            x_coord = pixel.pix_x
            y_coord = pixel.pix_y
            #Shift because x_coord,y_coord are the center points
            rectangle = plt.Rectangle((x_coord,y_coord),1,1, fc=cmap(SNR))
            #ax.add_patch(rectangle)
            patches.append(rectangle)
    colors = np.linspace(min(SNR_list),max(SNR_list),N)
    p = PatchCollection(patches, cmap=cmap)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    cbar = fig.colorbar(p, ax=ax)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Signal to Noise per Bin")
    cbar.set_label('Normalized Signal to Noise', rotation=270, labelpad=20)
    plt.savefig(file_dir+'/'+filename+".png")
    plt.clf()
    return None
#-------------------------------------------------#
#-------------------------------------------------#
# Bin Class Information
# Everything in here should be self-explanatory... if not let me know and I
# will most happily comment it! :)
class Bin:
    def __init__(self, number):
        self.bin_number = number
        self.pixels = []
        self.pixel_neighbors = []
        self.centroidx = [0] #So I can pass by value
        self.centroidy = [0]
        self.centroidx_prev = [0]
        self.centroidy_prev = [0]
        self.StN = [0]
        self.StN_prev = [0]
        self.Signal = [0]
        self.Noise = [0]
        self.Area = [0]
        self.Area_prev = [0]
        self.scale_length = [0]
        self.scale_length_prev = [0]
        self.successful = False
        self.WVT_successful = False
        self.avail_reassign = True #Can a pixel be reassigned to you?

    def add_pixel(self, Pixel):
        self.StN[0] = 0
        self.pixels.append(Pixel)
        self.Signal[0] += Pixel.Signal
        self.Noise[0] += Pixel.Noise
        if self.Noise[0] != 0:
            self.StN[0] = self.Signal[0]/(np.sqrt(self.Noise[0]))
        for neigh in Pixel.neighbors:
            if (neigh not in self.pixel_neighbors):
                self.pixel_neighbors.append(neigh)

    def clear_pixels(self):
        self.pixels = []
        self.StN[0] = 0
        self.Signal[0] = 0
        self.Noise[0] = 0
        self.successful = False
        self.WVT_successful = False
        self.avail_reassign = True

    def remove_pixel(self, Pixel):
        self.pixels.remove(Pixel)
        self.StN[0] = 0
        self.Signal[0] -= Pixel.Signal
        self.Noise[0] -= Pixel.Noise
        if self.Noise[0] != 0:
            self.StN[0] = self.Signal[0]/(np.sqrt(self.Noise[0]))

    def success(self):
        self.successful = True

    def WVT_success(self):
        self.WVT_successful = True

    def availabilty(self):
        self.avail_reassign = False

    def update_StN_prev(self):
        self.StN_prev[0] = self.StN[0]

    def CalcCentroid(self):
        self.centroidx_prev[0] = self.centroidx[0]
        self.centroidy_prev[0] = self.centroidy[0]
        self.centroidx[0] = 0
        self.centroidy[0] = 0
        n_cent = 0
        for pixel in self.pixels:
            self.centroidx[0] += pixel.pix_x
            self.centroidy[0] += pixel.pix_y
            n_cent += 1
        self.centroidx[0] *= (1/n_cent)
        self.centroidy[0] *= (1/n_cent)

    def CalcArea(self,pixel_length):
        self.Area_prev[0] = self.Area[0]
        self.Area[0] = 0
        self.Area[0] += pixel_length**2*len(self.pixels)

    def CalcScaleLength(self,stn_target):
        self.scale_length_prev[0] = self.scale_length[0]
        self.scale_length[0] = 0
        self.scale_length[0] += np.sqrt((self.Area[0]/np.pi)*(stn_target/self.StN[0]))
#-------------------------------------------------#
#-------------------------------------------------#
# Pixel Class Information
# Ditto à propos the documentation for this class
class Pixel:
    def __init__(self, number, pix_x, pix_y, signal, noise):
        self.pix_number = number
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.Signal = signal
        self.Noise = noise
        self.StN = 0
        if self.Noise != 0:
            self.StN = self.Signal/np.sqrt(self.Noise)
        self.neighbors = []
        self.neighbors_x = []
        self.neighbors_y = []
        self.assigned_to_bin = False
        self.assigned_bin = None
    def add_neighbor(self, pixel,x_pixel,y_pixel):
        self.neighbors.append(pixel)
        self.neighbors_x.append(x_pixel)
        self.neighbors_y.append(y_pixel)
    def add_to_bin(self,bin):
        self.assigned_to_bin = True
        self.assigned_bin = bin
    def clear_bin(self):
        self.assigned_to_bin = False
        self.assigned_bin = None
#-------------------------------------------------#
#-------------------------------------------------#
# READ IN
# we first must read in our data from the Chandra file
# pixel (x,y) are CENTER coordinates!
#   parameters:
#       fits_file = fits file in string format
#       image_fits = fits image file in string format
#       image_coord = reg file containing center of box of interest and sizes
#       exposure_map = exposure map file in string format
def read_in(image_fits,exposure_map = 'none'):
    #Collect Pixel Data
    hdu_list = fits.open(image_fits, memmap=True)
    counts = hdu_list[0].data
    y_len = counts.shape[0]
    x_len = counts.shape[1]
    hdu_list.close()
    x_min = 0; y_min = 0;
    x_max = x_len; y_max = y_len;
    if exposure_map.lower() != 'none':
        hdu_list = fits.open(exposure_map, memmap=True)
        exposure = hdu_list[0].data
        hdu_list.close()
        exposure_time = float(hdu_list[0].header["TSTOP"]) - float(hdu_list[0].header["TSTART"])
    #Currently Do not bother reading background information
    '''bkg_hdu = fits.open(bkg_image_fits, memmap=True)
    bkg_counts = bkg_hdu[0].data
    bkg_hdu.close()
    avg_bkg_counts = np.mean(bkg_counts)
    bkg_sigma = np.std(bkg_counts)'''
    Pixels = []
    pixel_count = 0
    for col in range(int(x_len)):
        for row in range(int(y_len)):
            if exposure_map.lower() == 'none':
                flux = counts[row][col]
                vari = counts[row][col]
            else:
                flux = counts[row][col]/(exposure[row][col]*exposure_time) #- avg_bkg_counts/(exposure[row][col]*exposure_time)
                vari = counts[row][col]/(exposure[row][col]**2*exposure_time**2) #+ bkg_sigma
            Pixels.append(Pixel(pixel_count,x_min+col,y_min+row,flux,vari)) #Bottom Left Corner!
            pixel_count += 1
    #print("We have "+str(pixel_count)+" Pixels! :)")
    return Pixels, x_min, x_max, y_min, y_max
#-------------------------------------------------#
#-------------------------------------------------#
# CALCULATE NearestNeighbors -- REALLY AN ADJACENCY LIST
#   http://scikit-learn.org/stable/modules/neighbors.html
#   parameters:
#       pixel_list - list of pixel objects
def Nearest_Neighbors(pixel_list):
    print("Running Nearest Neighbor Algorithm")
    xvals = []
    yvals = []
    num_neigh = 50
    for pixel in pixel_list:
        xvals.append(pixel.pix_x)
        yvals.append(pixel.pix_y)
    X = np.column_stack((xvals,yvals))
    nbrs = NearestNeighbors(n_neighbors=num_neigh, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    pix_num = 0
    for pixel in pixel_list:
        for j in range(num_neigh-1):
            if distances[pix_num][j+1] == 1:
                index = indices[pix_num][j+1]
                pixel.add_neighbor(pixel_list[index],xvals[index],yvals[index])
            else:
                pass #not adjacent
        pix_num += 1
    print("Finished Nearest Neighbor Algorithm")
    return None
#-------------------------------------------------#
#-------------------------------------------------#
# CALCULATE CLOSEST NODE and Distances
#   parameters:
#       Bin_current = current bin object
#       unassigned_pixels - list of indices of unassigned pixels
def dist(p1x,p1y,p2x,p2y):
    return np.sqrt((p1x-p2x)**2+(p1y-p2y)**2)
def closest_node(Bin_current,unassigned_pixels):
    closest_val = 1e16 #just some big number
    p1x = Bin_current.centroidx[0]
    p1y = Bin_current.centroidy[0]
    for pix_neigh in Bin_current.pixel_neighbors:
        if pix_neigh.assigned_to_bin == False: #Dont bother with already assigned pixels!
            p2x = pix_neigh.pix_x
            p2y = pix_neigh.pix_y
            new_dist = dist(p1x,p1y,p2x,p2y)
            if new_dist < closest_val:
                closest_val = new_dist
                closest_pixel = pix_neigh
    if closest_val == 1e16:
        #None of the neighbors work so just pick a random unassigned pixel
        dist_list = [dist(p1x,p1y,pixel.pix_x,pixel.pix_y) for pixel in unassigned_pixels]
        closest_pixel = unassigned_pixels[dist_list.index(min(dist_list))]
    return closest_pixel
#-------------------------------------------------#
#-------------------------------------------------#
# CALCULATE Adjacency
#   parameters:
#       current_bin - current bin object
#       closest_node - closest pixel to bin
def adjacency(current_bin,closest_node):
    if closest_node in current_bin.pixel_neighbors:
        return True
    else:
        return False
#-------------------------------------------------#
#-------------------------------------------------#
# CALCULATE Roundness
#   parameters:
#       current_bin - current bin object
#       closest_pixel - closest pixel to bin
#       pixel_length - twice the radius of a pixel!
def Roundness(current_bin,closest_pixel,pixel_length):
    pixel_list = current_bin.pixels
    n = len(pixel_list)+1 #Including new
    rad_equiv = np.sqrt(n/np.pi)*(pixel_length)
    cen_x_new = 0 ; cen_y_new = 0
    xvals = [pixel.pix_x for pixel in current_bin.pixels]
    yvals = [pixel.pix_y for pixel in current_bin.pixels]
    cen_x_new = (sum(xvals)+closest_pixel.pix_x)/n
    cen_y_new = (sum(yvals)+closest_pixel.pix_y)/n
    dists = []
    for i in range(n-1):
        dists.append(np.sqrt((xvals[i] - cen_x_new)**2+(yvals[i] - cen_y_new)**2))
    dists.append(np.sqrt((closest_pixel.pix_x-cen_x_new)**2+(closest_pixel.pix_y-cen_y_new)**2))
    rad_max = max(dists) # maximum distance between the centroid of the bin and any of the bin pixels
    roundness = rad_max/rad_equiv - 1.
    return roundness
#-------------------------------------------------#
#-------------------------------------------------#
# CALCULATE StN_Potential
#   parameters:
#       Current_bin - current bin object
#       closest_pix - closest pixel to bin
def Potential_SN(Current_bin,closest_pix):
    Current_bin.add_pixel(closest_pix)
    new_StN = Current_bin.StN[0]
    Current_bin.remove_pixel(closest_pix)
    return new_StN
#-------------------------------------------------#
#-------------------------------------------------#
# Create bin information for chandra
#   prameters:
#       Bins - list of final bins
#       min_x - minimum pixel x value
#       min_y - minimum pixel y value
#       output_directory - directory where txt file will be located
#       filename - name of text file
def Bin_data(Bins,missing_pixels,min_x,min_y, output_directory, filename):
    Bins.sort(key=lambda bin: bin.bin_number)
    file = open(output_directory+'/'+filename+'.txt',"w+")
    file.write("This text file contains information necessary for chandra to bin the pixels appropriately for image.fits \n")
    file.write("pixel_x pixel_y bin \n")
    file2 = open(output_directory+'/'+filename+'_bins.txt','w+')
    file2.write("Bin data for paraview script to plot Weighted Voronoi Diagram \n")
    file2.write("centroidx centroidy weight \n")
    binCount = 0
    for bin in Bins:
        for pixel in bin.pixels:
            file.write(str(pixel.pix_x-min_x)+" "+str(pixel.pix_y-min_y)+" "+str(binCount)+' \n')
        file2.write(str(bin.centroidx[0])+" "+str(bin.centroidy[0])+" "+str(bin.scale_length[0])+" \n")
        binCount += 1
    file.close()
    file2.close()
    return None
#-------------------------------------------------#
#-------------------------------------------------#
# Find bin number given pixel
#   parameters:
#       binList - list of bin objects
#       currentPixel  - current pixel object
def bin_num_pixel(binList,currentPixel):
    bin_number_interest = None
    for binNumber in range(len(binList)):
        if currentPixel in binList[binNumber].pixels:
            bin_number_interest = binNumber
            break
    return bin_number_interest
#-------------------------------------------------#
#-------------------------------------------------#
# Assign pixels with no StN to closest bin_num
#   parameters:
#       bin_list - list of bins
#       pixels - list of pixels
def assigned_missing_pixels(pixels):
    #get all pixels in domain but not already assigned
    pos_x = [pix.pix_x for pix in pixels]
    pos_y = [pix.pix_y for pix in pixels]
    already_pixel = {}
    for i in range(len(pos_x)):
        already_pixel[(pos_x[i],pos_y[i])] = True
    x_range = [i for i in range(min(pos_x), max(pos_x)+1)]
    y_range = [i for i in range(min(pos_y), max(pos_y)+1)]
    Pixels_unbinned = []
    unbinned_num = max([pix.pix_number for pix in pixels])+1
    for x in x_range:
        for y in y_range:
            if (x,y) not in already_pixel.keys():
                new_pixel = Pixel(unbinned_num,x,y,0)
                Pixels_unbinned.append(new_pixel)
                unbinned_num += 1
            else:
                pass #Pixel already binned
    print("We have "+str(len(Pixels_unbinned))+" unbinned pixels")
    return Pixels_unbinned
#-------------------------------------------------#
#-------------------------------------------------#
# Reassign pixels that were in bins that were too small (StN)
#   parameters:
#       bin - current bin
#       bins_successful - list of potential new bins
#       sucessful_centroids -- centroids of successful bins
def reassign_pixels(bin,bins_successful,sucessful_centroids):
    for pixel in bin.pixels:
        pixel.clear_bin()
        potential_bins = bins_successful[:]
        potential_centroids = sucessful_centroids[:]
        while pixel.assigned_to_bin == False:
            dists = [dist(pixel.pix_x,pixel.pix_y,centx,centy) for (centx,centy) in potential_centroids]
            closest_bin_index = dists.index(min(dists))
            closest_bin = potential_bins[closest_bin_index]
            if closest_bin.availabilty == False and len(potential_centroids) > 1:
                del potential_centroids[closest_bin_index]
                potential_bins.remove(closest_bin)
            else:
                pixel.add_to_bin(closest_bin)
                closest_bin.add_pixel(pixel)
    return None
#-------------------------------------------------#
#-------------------------------------------------#
# Bin-Accretion Algorithm
#   parameters:
#       Pixels - list of unique pixels
#       pixel_length - physical size of pixel
#       stn_target - Target value of Signal-to_noise
#       roundness_crit - Critical roundness value (upper bound)
def Bin_Acc(Pixels,pixel_length,stn_target,roundness_crit):
    #step 1:setup list of bin objects
    print("Starting Bin Accretion Algorithm")
    unassigned_pixels = Pixels[:]
    binCount = 0
    Bin_list = []
    bins_successful = []
    criteria_a = False; criteria_b = False; criteria_c = False
    StN_list_pixels = [pixel.StN for pixel in Pixels]
    max_StN_ind = StN_list_pixels.index(max(StN_list_pixels))
    max_StN_pix = Pixels[max_StN_ind]
    Current_bin = Bin(binCount)
    Bin_list.append(Current_bin)
    Current_bin.add_pixel(max_StN_pix)
    Current_bin.CalcCentroid()
    max_StN_pix.add_to_bin(binCount)
    unassigned_pixels.remove(max_StN_pix)
    closest_pix = None
    closest_not_in_pix = None
    while len(unassigned_pixels) != 0:
        criteria_a = True; criteria_b = True; criteria_c = True;
        while (criteria_a == True and criteria_b == True and criteria_c == True) and len(unassigned_pixels) != 0:
            closest_pix = closest_node(Current_bin,unassigned_pixels)
            criteria_a = adjacency(Current_bin,closest_pix)
            criteria_b = True if (Roundness(Current_bin,closest_pix,pixel_length) < roundness_crit) else False
            criteria_c = True if (Potential_SN(Current_bin,closest_pix) < 0.75*stn_target) else False
            if (criteria_a == True and criteria_b == True and criteria_c == True):
                Current_bin.add_pixel(closest_pix)
                Current_bin.CalcCentroid()
                closest_pix.add_to_bin(binCount)
                unassigned_pixels.remove(closest_pix)
            else:
                closest_not_in_pix = closest_pix
        if (Current_bin.StN[0] > 0.5*stn_target):
            Current_bin.success()
            bins_successful.append(Current_bin)
        if len(unassigned_pixels) == 0:
            break #All pixels assigned so dont create a new bin. that would be silly
        else:
            binCount += 1
            Current_bin = Bin(binCount)
            Bin_list.append(Current_bin)
            Current_bin.add_pixel(closest_not_in_pix)
            Current_bin.CalcCentroid()
            closest_not_in_pix.add_to_bin(binCount)
            unassigned_pixels.remove(closest_not_in_pix)
    print("Reassigning unsuccessful bins")
    for bin in bins_successful:
        bin.CalcCentroid()
        bin.CalcArea(pixel_length)
        bin.CalcScaleLength(stn_target)
    sucessful_centroids = [(bin.centroidx[0],bin.centroidy[0]) for bin in bins_successful]
    for bin in Bin_list:
        if bin.successful == False:
            reassign_pixels(bin,bins_successful,sucessful_centroids)
            bin.clear_pixels()
        else:
            bin.CalcCentroid()
            bin.CalcArea(pixel_length)
            bin.CalcScaleLength(stn_target)
    for bin in bins_successful: bin.update_StN_prev()
    print("Completed Bin Accretion Algorithm")
    print("There are a total of "+str(len(bins_successful)+1)+" bins!")
    return bins_successful
#-------------------------------------------------#
#-------------------------------------------------#
# Converged Criteria
#   parameters:
#       Bins - list of old bins
#       tol - tolerance level for distance convergence (usually 1e-16 ==> machine precision)
def converged_met(Bins,tol):
    True_count = 0
    for bin in Bins:
        StN_old = bin.StN_prev[0]
        StN_new = bin.StN[0]
        if abs(StN_new-StN_old)/StN_old < tol:
            True_count += 1
        else:
            pass
    if True_count/len(Bins) > 0.8:
        return True
    else:
        return False
#-------------------------------------------------#
#-------------------------------------------------#
# Rebin Pixels
#   parameters:
#       binList - list of current bins
#       pixel_list - list of all pixels
#       pixel_length - twice the radius of a pixel
#       stn_target - target Signal-to-Noise
def Rebin_Pixels(binList,pixel_list,pixel_length,stn_target):
    for bin in binList:
        bin.clear_pixels()
    WVT_successful_bins = []
    for pixel in pixel_list:
        pixel.clear_bin()
        distances_bins = [dist(pixel.pix_x,pixel.pix_y,bin.centroidx_prev[0],bin.centroidy_prev[0])/bin.scale_length_prev[0] for bin in binList]
        closest_bin = binList[distances_bins.index(min(distances_bins))]
        pixel.add_to_bin(closest_bin)
        closest_bin.add_pixel(pixel)
        if closest_bin.StN[0] > 0 and closest_bin.WVT_successful == False:
            WVT_successful_bins.append(closest_bin)
            closest_bin.WVT_success()
    sucessful_centroids = [(bin.centroidx[0],bin.centroidy[0]) for bin in WVT_successful_bins]
    for bin in binList:
        if bin not in WVT_successful_bins:
            reassign_pixels(bin,WVT_successful_bins,sucessful_centroids)
    for bin in WVT_successful_bins:
        bin.CalcCentroid()
        bin.CalcArea(pixel_length)
        bin.CalcScaleLength(stn_target)
    return WVT_successful_bins
#-------------------------------------------------#
#-------------------------------------------------#
# Subroutine to calculate the weighted voronoi tessellations
# Using the Bin Accretion algorithm as an initial condition
#   parameters:
#       Bin_list_init - list of bin accretion bins
#       Pixel_Full - All of the pixels including ones missing
#       stn_target - Target Signal-to-Noise value
#       tol - tolerance Level
#       pixel_length - twice the pixel radius
#       image_dir - directory for images
def WVT(Bin_list_init,Pixel_Full,stn_target,tol,pixel_length,image_dir):
    print("Beginning WVT")
    Bin_list_prev = Bin_list_init[:]
    converged = False
    its_to_conv = 0
    while converged == False and its_to_conv<5:
        print("We are on step "+str(its_to_conv+1))
        bins_with_SN = Rebin_Pixels(Bin_list_prev,Pixel_Full,pixel_length,stn_target)[:]
        converged = converged_met(bins_with_SN,tol)
        Bin_list_prev = bins_with_SN[:]
        for bin in bins_with_SN: bin.update_StN_prev()
        bin_SN_List = [bin.StN[0] for bin in bins_with_SN]
        plt.hist(bin_SN_List)
        plt.ylabel("Number of Bins")
        plt.xlabel("Signal-to-Noise")
        its_to_conv += 1
        plt.savefig(image_dir+'/histograms/iteration_'+str(its_to_conv)+".png")
        plt.clf()
    print("Completed WVT in "+str(its_to_conv)+" step(s)!!")
    print("There are a total of "+str(len(bins_with_SN)+1)+" bins!")
    return bins_with_SN
#-------------------------------------------------#
#-------------------------------------------------#

def wvt_main(inputs,base,output_name):
    os.chdir(base)
    if os.path.isdir(base+'/histograms') == False:
        os.mkdir(base+'/histograms')
    else:
        for graph in os.listdir(base+'/histograms'):
            if graph.endswith(".png"):
                os.remove(os.path.join(base+'/histograms', graph))
    Pixels = []
    pixel_size = inputs['pixel_radius']*2
    print("#----------------Algorithm Part 1----------------#")
    print(os.getcwd())
    Pixels,min_x,max_x,min_y,max_y = read_in(inputs['image_fits'],inputs['exposure_map'])
    Nearest_Neighbors(Pixels)
    Init_bins = Bin_Acc(Pixels,pixel_size,inputs['stn_target'],inputs['roundness_crit'])
    plot_Bins(Init_bins,min_x,max_x,min_y,max_y,inputs['stn_target'],base,"bin_acc")
    print("#----------------Algorithm Part 2----------------#")
    Final_Bins = WVT(Init_bins,Pixels,inputs['stn_target'],inputs['tol'],pixel_size,base)
    print("#----------------Algorithm Complete--------------#")
    plot_Bins(Final_Bins,min_x,max_x,min_y,max_y,inputs['stn_target'],base,"final")
    #plot_Bins_SN(Final_Bins,min_x,max_x,min_y,max_y,inputs['image_dir'],"WVT_SN")
    Bin_data(Final_Bins,Pixels,min_x,min_y,base,output_name)
    print("#----------------Information Stored--------------#")
