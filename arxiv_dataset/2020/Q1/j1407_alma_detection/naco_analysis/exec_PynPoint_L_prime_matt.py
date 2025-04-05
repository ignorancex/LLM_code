# -*- coding: utf-8 -*-

__author__ = "Alexander Bohn"

"""
Executor of the PynPoint pipeline for J1407 in L_prime band
"""

import os

import numpy as np

from pynpoint import Pypeline, FitsReadingModule, FitsWritingModule, FlatCalibrationModule, \
    BadPixelSigmaFilterModule, CropImagesModule, AngleCalculationModule, \
    RemoveLastFrameModule, RemoveLinesModule, FitCenterModule, ShiftImagesModule, \
    StackAndSubsetModule, DitheringBackgroundModule, DerotateAndStackModule, PSFpreparationModule, \
    PcaPsfSubtractionModule, ReplaceBadPixelsModule, DarkCalibrationModule, \
    RemoveStartFramesModule, FrameSelectionModule, ClassicalADIModule, StarAlignmentModule, Hdf5WritingModule, \
    BadPixelMapModule

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

iwa = 4
pixscale = 0.02718
stack_number = 10
norm = False
pca_numbers = np.arange(100)

print("\nRunning PynPoint on target:\n%s\n" % (target))

# Define Directories
working_place_in = os.path.join("/Users/alex/workplace/targets/", target, epoch, instrument, filt, "science")
input_place_in = os.path.join(path, target, epoch, instrument, filt, exptime)
output_place_in = os.path.join(input_place_in, "Results")

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

# --- read Science data

science_reading = FitsReadingModule(name_in="science_reading",
                                    input_dir=os.path.join(input_place_in, "OBJECT"),
                                    image_tag="im_arr")

# ---- read flats

read_flat = FitsReadingModule(name_in="flat_reading",
                              input_dir=os.path.join(input_place_in, "FLAT_SKY"),
                              image_tag="flat_arr")

# ---- read dark images

read_dark_flat = FitsReadingModule(name_in="dark_reading_flat",
                                   input_dir=os.path.join(input_place_in, "DARK", "1024_1024"),
                                   image_tag="dark_arr_flat")

# ---- dark subtraction flat

dark_subtraction_flat = DarkCalibrationModule(name_in="dark_subtraction_flat",
                                              image_in_tag="flat_arr",
                                              dark_in_tag="dark_arr_flat",
                                              image_out_tag="flat_arr_sub")

write_flat_arr_sub = FitsWritingModule(file_name="master_flat_arr.fits",
                                       name_in="write_flat_arr_sub",
                                       data_tag="flat_arr_sub")

# --- dark reading

dark_reading = FitsReadingModule(name_in="dark_reading",
                                 input_dir=os.path.join(input_place_in, "DARK", "512_520"),
                                 image_tag="dark_arr")

# ---- remove last frame of science images

remove_last_frame_science = RemoveLastFrameModule(name_in="remove_last_frame_science",
                                                  image_in_tag="im_arr",
                                                  image_out_tag="im_arr_last_frame_removed")

# ---- remove last frame of science images

remove_start_frames = RemoveStartFramesModule(frames=5,
                                              name_in="remove_start_frames",
                                              image_in_tag="im_arr_last_frame_removed",
                                              image_out_tag="im_arr_good")

# ---- cut science images to resolution of (511,511)

cut_im_arr = RemoveLinesModule(lines=(0, 1, 4, 5),
                               name_in="cut_im_arr",
                               image_in_tag="im_arr_good",
                               image_out_tag="im_arr_cut")

write_im_arr_cut = FitsWritingModule(file_name="im_arr_cut.fits",
                                     name_in="write_im_arr_cut",
                                     data_tag="im_arr_cut")

# ---- cut dark images to resolution of (511,511)

cut_dark_arr = RemoveLinesModule(lines=(0, 1, 4, 5),
                                 name_in="cut_dark_arr",
                                 image_in_tag="dark_arr",
                                 image_out_tag="dark_arr_cut")

write_dark_arr_cut = FitsWritingModule(file_name="dark_arr_cut.fits",
                                       name_in="write_dark_arr_cut",
                                       data_tag="dark_arr_cut")

# ---- cut flat images to resolution of (511,511)

cut_flat_arr_sub = CropImagesModule(size=im_size_1 * pixscale,
                                    center=None,
                                    name_in="cut_flat_arr_sub",
                                    image_in_tag="flat_arr_sub",
                                    image_out_tag="flat_arr_cut")

write_flat_arr_cut = FitsWritingModule(file_name="flat_arr_cut.fits",
                                       name_in="write_flat_arr_cut",
                                       data_tag="flat_arr_cut")

# ---- subtract dark from science

dark_subtraction = DarkCalibrationModule(name_in="dark_calibration",
                                         image_in_tag="im_arr_cut",
                                         dark_in_tag="dark_arr_cut",
                                         image_out_tag="im_arr_cut_dark_sub")

write_im_arr_cut_dark_sub = FitsWritingModule(file_name="im_arr_cut_dark_sub.fits",
                                              name_in="write_im_arr_cut_dark_sub",
                                              data_tag="im_arr_cut_dark_sub")

# --- subtract flat from science

flat_subtraction_science = FlatCalibrationModule(name_in="flat_subtraction_science",
                                                 image_in_tag="im_arr_cut_dark_sub",
                                                 flat_in_tag="flat_arr_cut",
                                                 image_out_tag="im_arr_clean")

write_im_arr_clean = FitsWritingModule(file_name="im_arr_clean.fits",
                                       name_in="write_im_arr_clean",
                                       data_tag="im_arr_clean")

# --- create bad pixel map

create_bp_map = BadPixelMapModule(name_in="create_bad_pixel_map",
                                  dark_in_tag="dark_arr_cut",
                                  flat_in_tag="flat_arr_cut",
                                  bp_map_out_tag="bp_map",
                                  dark_threshold=0.2,
                                  flat_threshold=0.2)

write_bp_map = FitsWritingModule(file_name="bp_map.fits",
                                 name_in="write_bp_map",
                                 data_tag="bp_map")

# --- run bad pixel cleaning on Science data

bp_cleaning_science_1 = ReplaceBadPixelsModule(name_in="bad_pixel_cleaning_1",
                                               image_in_tag="im_arr_clean",
                                               map_in_tag="bp_map",
                                               image_out_tag="tmp_im_arr_bp_clean",
                                               size=2,
                                               replace="mean")

bp_cleaning_science_2 = BadPixelSigmaFilterModule(name_in="bad_pixel_cleaning_2",
                                                  image_in_tag="tmp_im_arr_bp_clean",
                                                  image_out_tag="im_arr_bp_clean",
                                                  box=9,
                                                  sigma=5,
                                                  iterate=1)

write_im_arr_bp_clean = FitsWritingModule(file_name="im_arr_bp_clean.fits",
                                          name_in="write_im_arr_bp_clean",
                                          data_tag="im_arr_bp_clean")

# --- angle calculation

angle_calculation = AngleCalculationModule(instrument=instrument,
                                           name_in="angle_calculation",
                                           data_tag="im_arr_bp_clean")

# ---- background subtraction

background_subtraction = DitheringBackgroundModule(name_in="background_subtraction",
                                                   image_in_tag="im_arr_bp_clean",
                                                   image_out_tag="im_arr_bg_sub",
                                                   center=None,
                                                   cubes=None,
                                                   size=im_size_2 * pixscale,  # TODO
                                                   gaussian=4 * pixscale,  # TODO
                                                   subframe=20 * pixscale,
                                                   pca_number=50,
                                                   mask_star=15 * pixscale)

write_im_arr_bg_sub = FitsWritingModule(file_name="im_arr_bg_sub.fits",
                                        name_in="write_im_arr_bg_sub",
                                        data_tag="im_arr_bg_sub")

# --- align science data

star_alignment = StarAlignmentModule(name_in="align_science",
                                     image_in_tag="im_arr_bg_sub",
                                     ref_image_in_tag=None,
                                     image_out_tag="im_arr_aligned",
                                     interpolation="spline",
                                     accuracy=10,
                                     resize=None,
                                     num_references=10,
                                     subframe=40 * pixscale)

write_im_arr_aligned = FitsWritingModule(file_name="im_arr_aligned.fits",
                                         name_in="write_im_arr_aligned",
                                         data_tag="im_arr_aligned")

# --- center science images by a Gaussian fit

fit_center_science = FitCenterModule(name_in="fit_center_science",
                                     image_in_tag="im_arr_aligned",
                                     fit_out_tag="science_center_fit",
                                     mask_out_tag=None,
                                     method="mean",
                                     radius=20 * pixscale,
                                     sign="positive",
                                     model="gaussian",
                                     filter_size=None,
                                     guess=(-5, -7, 4, 4, 1500., 0., 0.))

shift_center_science = ShiftImagesModule(name_in="shift_center_science",
                                         image_in_tag="im_arr_aligned",
                                         image_out_tag="im_arr_centered",
                                         shift_xy="science_center_fit",
                                         interpolation="spline")

write_im_arr_centered = FitsWritingModule(file_name="im_arr_centered.fits",
                                          name_in="write_im_arr_centered",
                                          data_tag="im_arr_centered")

# --- cut images due to residuals from alignment

cut_im_arr_centered = CropImagesModule(size=im_size_3 * pixscale,
                                       center=None,
                                       name_in="cut_im_arr_centered",
                                       image_in_tag="im_arr_centered",
                                       image_out_tag="im_arr_centered_cut")

write_im_arr_centered_cut = FitsWritingModule(file_name="im_arr_centered_cut.fits",
                                              name_in="write_im_arr_centered_cut",
                                              data_tag="im_arr_centered_cut")

#  --- frame rejection

frame_selection_1 = FrameSelectionModule(name_in="frame_selection_1",
                                         image_in_tag="im_arr_centered_cut",
                                         selected_out_tag="tmp_im_arr_selected_1",
                                         removed_out_tag="im_arr_removed_1",
                                         index_out_tag=None,
                                         method="median",
                                         threshold=2.,
                                         fwhm=4 * pixscale,
                                         aperture=("annulus", 35 * pixscale, 45 * pixscale),
                                         position=(im_size_3 / 2, im_size_3 / 2, 5 * pixscale))

frame_selection_2 = FrameSelectionModule(name_in="frame_selection_2",
                                         image_in_tag="tmp_im_arr_selected_1",
                                         selected_out_tag="im_arr_selected",
                                         removed_out_tag="im_arr_removed_2",
                                         index_out_tag=None,
                                         method="median",
                                         threshold=2.,
                                         fwhm=4 * pixscale,
                                         aperture=("circular", 5 * pixscale),
                                         position=(im_size_3 / 2, im_size_3 / 2, 5 * pixscale))

write_im_arr_selected = FitsWritingModule(file_name="im_arr_selected.fits",
                                          name_in="write_im_arr_selected",
                                          data_tag="im_arr_selected")

wrtie_im_arr_selected_hdf5 = Hdf5WritingModule(file_name="im_arr_selected.hdf5",
                                               name_in="wrtie_im_arr_selected_hdf5",
                                               output_dir=working_place_in,
                                               tag_dictionary={"im_arr_selected": "im_arr_selected"},
                                               keep_attributes=True,
                                               overwrite=True)

write_im_arr_removed_1 = FitsWritingModule(file_name="im_arr_removed_1.fits",
                                           name_in="write_im_arr_removed_1",
                                           data_tag="im_arr_removed_1")

write_im_arr_removed_2 = FitsWritingModule(file_name="im_arr_removed_2.fits",
                                           name_in="write_im_arr_removed_2",
                                           data_tag="im_arr_removed_2")
# ---- derotate and stack images

derotate_median = DerotateAndStackModule(name_in="derotate_images",
                                         image_in_tag="im_arr_selected",
                                         image_out_tag="im_arr_derotated_median",
                                         derotate=True,
                                         stack="median")

write_im_arr_derotated_median = FitsWritingModule(file_name="%s_%s_derotated_median.fits" % (target, filt),
                                                  name_in="write_im_arr_derotated_median",
                                                  data_tag="im_arr_derotated_median")

# --- perform cADI

classical_ADI = ClassicalADIModule(threshold=None,
                                   nreference=None,
                                   residuals="median",
                                   extra_rot=0.,
                                   name_in="classical_ADI",
                                   image_in_tag="im_arr_selected",
                                   res_out_tag="res_rot_cADI",
                                   stack_out_tag="res_median_cADI")

write_res_median_cADI = FitsWritingModule(file_name="%s_%s_res_median_cADI.fits" % (target, filt),
                                          name_in="write_res_median_cADI",
                                          data_tag="res_median_cADI")

# --- stack images for PCA

stack_images_pca = StackAndSubsetModule(name_in="stack_images_pca",
                                        image_in_tag="im_arr_selected",
                                        image_out_tag="im_arr_stacked",
                                        stacking=stack_number)

#  --- ADI+PCA and median combine

prepare_data_pca = PSFpreparationModule(name_in="psf_preparation_pca",
                                        image_in_tag="im_arr_stacked",
                                        image_out_tag="im_arr_prep_pca",
                                        mask_out_tag="mask_arr",
                                        norm=norm,
                                        resize=None,
                                        cent_size=None,  # iwa*pixscale,
                                        edge_size=(im_size_3 / 2.) * pixscale)

write_im_arr_prep_pca = FitsWritingModule(file_name="%s_%s_arr_prep_pca.fits" % (target, filt),
                                          name_in="write_im_arr_prep_pca",
                                          data_tag="im_arr_prep_pca")

adi_pca_median = PcaPsfSubtractionModule(pca_numbers=pca_numbers,
                                         name_in="adi_pca_median",
                                         images_in_tag="im_arr_prep_pca",
                                         reference_in_tag="im_arr_prep_pca",
                                         res_mean_tag="res_mean",
                                         res_median_tag="res_median",
                                         res_arr_out_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         extra_rot=0.,
                                         subtract_mean=True)

write_res_mean = FitsWritingModule(file_name="%s_%s_res_mean_norm_%s_pca_%s.fits" % (target, filt, norm, stack_number),
                                   name_in="write_res_mean",
                                   data_tag="res_mean")

write_res_median = FitsWritingModule(
    file_name="%s_%s_res_median_norm_%s_pca_%s.fits" % (target, filt, norm, stack_number),
    name_in="write_res_median",
    data_tag="res_median")

# Adding modules
pipeline.add_module(science_reading)
pipeline.add_module(read_flat)
pipeline.add_module(read_dark_flat)
pipeline.add_module(dark_subtraction_flat)
pipeline.add_module(write_flat_arr_sub)
pipeline.add_module(dark_reading)
pipeline.add_module(remove_last_frame_science)
pipeline.add_module(remove_start_frames)
pipeline.add_module(cut_im_arr)
# pipeline.add_module(write_im_arr_cut)
pipeline.add_module(cut_dark_arr)
# pipeline.add_module(write_dark_arr_cut)
pipeline.add_module(cut_flat_arr_sub)
# pipeline.add_module(write_flat_arr_cut)
pipeline.add_module(dark_subtraction)
# pipeline.add_module(write_im_arr_cut_dark_sub)
pipeline.add_module(flat_subtraction_science)
# pipeline.add_module(write_im_arr_clean)
pipeline.add_module(create_bp_map)
# pipeline.add_module(write_bp_map)
pipeline.add_module(bp_cleaning_science_1)
pipeline.add_module(bp_cleaning_science_2)
# pipeline.add_module(write_im_arr_bp_clean)
pipeline.add_module(angle_calculation)
pipeline.add_module(background_subtraction)
# pipeline.add_module(write_im_arr_bg_sub)
pipeline.add_module(star_alignment)
# pipeline.add_module(write_im_arr_aligned)
pipeline.add_module(fit_center_science)
pipeline.add_module(shift_center_science)
# pipeline.add_module(write_im_arr_centered)
pipeline.add_module(cut_im_arr_centered)
# pipeline.add_module(write_im_arr_centered_cut)
pipeline.add_module(frame_selection_1)
pipeline.add_module(frame_selection_2)
pipeline.add_module(write_im_arr_selected)
pipeline.add_module(wrtie_im_arr_selected_hdf5)
pipeline.add_module(write_im_arr_removed_1)
pipeline.add_module(write_im_arr_removed_2)
pipeline.add_module(derotate_median)
pipeline.add_module(write_im_arr_derotated_median)
pipeline.add_module(classical_ADI)
pipeline.add_module(write_res_median_cADI)
pipeline.add_module(stack_images_pca)
pipeline.add_module(prepare_data_pca)
# pipeline.add_module(write_im_arr_prep_pca)
pipeline.add_module(adi_pca_median)
pipeline.add_module(write_res_mean)
pipeline.add_module(write_res_median)

pipeline.run()
