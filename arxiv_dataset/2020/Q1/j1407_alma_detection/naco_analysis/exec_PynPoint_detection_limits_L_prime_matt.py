# -*- coding: utf-8 -*-

__author__ = "Alexander Bohn"

"""
Executor of the PynPoint pipeline for J1407 in L_prime band
"""

import os

from pynpoint import Pypeline, StackAndSubsetModule, ContrastCurveModule, TextWritingModule

# --- Parameters

target = "J1407"
epoch = "2019-03-01"
instrument = "NACO"
exptime = "0_2"
filt = "L_prime"
path = "/Users/alex/data/targets/"
im_size_1 = 511
im_size_2 = 115
im_size_3 = 101

# iwa = 4
pixscale = 0.02718
stack_number = 10
norm = False
pca_numbers = 1

print("\nRunning PynPoint on target:\n%s\n" % (target))

# Define Directories
working_place_in = os.path.join("/Users/alex/workplace/targets/", target, epoch, instrument, filt, "science")
input_place_in = os.path.join(path,target,epoch,instrument,filt,exptime)
output_place_in = os.path.join(input_place_in,"Results","detection_limits")

# Create Working directory for this target
if not os.path.isdir(working_place_in):
    print("Creating directory: %s" % (working_place_in))
    os.makedirs(working_place_in)

# Create directory for results
if not os.path.isdir(output_place_in):
    print("Creating directory: %s" % (output_place_in))
    os.makedirs(output_place_in)

pipeline = Pypeline(working_place_in,
                    input_place_in,
                    output_place_in)

# --- stack images for detection limits

stack_images_pca = StackAndSubsetModule(name_in="stack_images_pca",
                                        image_in_tag="im_arr_selected",
                                        image_out_tag="im_arr_stacked",
                                        stacking=stack_number)

#  --- determine detction limits for several PAs and separations
detection_limits = ContrastCurveModule(name_in="detection_limits",
                                       image_in_tag="im_arr_stacked",
                                       psf_in_tag="im_arr_stacked",
                                       contrast_out_tag="detection_limits",
                                       separation=(5*pixscale,im_size_3/2.*pixscale,2.*pixscale),
                                       angle=(0.,360.,60.),
                                       threshold=("sigma", 5.),
                                       accuracy=1e-1,
                                       psf_scaling=1.,
                                       aperture=2.*pixscale,
                                       ignore=False,
                                       pca_number=pca_numbers,
                                       norm=norm,
                                       cent_size=None,
                                       edge_size=(im_size_3 / 2.) * pixscale,
                                       extra_rot=0.,
                                       residuals="median",
                                       snr_inject=100)

write_detection_limits = TextWritingModule(file_name="detection_limits.dat",
                                           name_in="write_detection_limits",
                                           data_tag="detection_limits")


# Adding modules
pipeline.add_module(stack_images_pca)
pipeline.add_module(detection_limits)
pipeline.add_module(write_detection_limits)

pipeline.run()