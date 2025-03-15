import os
from astropy import units as u

import BD_wrapper.BD_wrapper as bdw


#  dirpath to the input directory
dirpath_inp = os.path.join(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))), 'data')
#  dirpath of the output directory
dirpath_out = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'results')

#  check if dirpath_out already exists, if not create it
if not os.path.exists(dirpath_out):
    os.makedirs(dirpath_out)

table_name = 'input_data'
suffix = ''


def main():
    b = bdw.BayesianDistance()

    #  which version of the BDC to use; set to '1.0' to use BDC v1
    b.version = '2.4'

    #  specify number of CPUs used for multiprocessing
    b.use_ncpus = 1

    #  save temporary files produced by the BDC in dirpath_out
    b.save_temporary_files = True

    #  produce a plot of the distance probability density functions produced by the BDC
    b.plot_probability = True

    #
    #  define input/output tables
    #

    b.path_to_input_table = os.path.join(
        dirpath_inp,
        '{}.dat'.format(table_name))
    b.path_to_output_table = os.path.join(
        dirpath_out,
        '{}_distance_results{}.dat'.format(table_name, suffix))
    #  specify the table format
    b.table_format = 'ascii'

    #  specify the column names of the Galactic Longitude, Galactic Latitude and VLSR values in the input table. If there are no column names, this can also be specified in terms of the column number, i.e. colnr_lon = 0, colnr_lat = 1, etc.
    b.colname_lon = 'GLON'
    b.colname_lat = 'GLAT'
    b.colname_vel = 'VLSR'
    #  use either e_VLSR or vel_disp for e_vel, whichever value is higher
    b.colname_e_vel = ['e_VLSR', 'vel_disp']
    b.colname_vel_disp = 'vel_disp'
    b.colname_name = 'Name'

    #
    #  set weights for the priors
    #

    #  set weight for spiral arm prior (default: 0.85)
    b.prob_sa = 0.5
    #  set weight for maser parallax prior (default: 0.15)
    b.prob_ps = 0.15
    #  set weight for Galactic Latitude prior (default: 0.85)
    b.prob_gl = 0.5
    #  set weight for kinematic distance prior (default: 0.85)
    b.prob_kd = 0.85

    #
    #  settings for the KDA prior
    #

    #  use literature distance solutions to inform the prior for the kinematic distance ambiguity
    b.check_for_kda_solutions = True
    #  specify tables from the KDA_info directory that should be used for the KDA prior; by default all tables are used
    b.kda_info_tables = []
    #  specify tables from the KDA_info directory that should be excluded for the KDA prior; by default no tables are excluded
    b.exclude_kda_info_tables = []

    #  Uncomment the following line to use a given KDA solution instead of estimating it; kda_weight defines the weight given to this KDA solution with 0 meaning no weight and 1 being the maximum weight
    # b.colname_kda = 'KDA'
    # b.kda_weight = 0.75

    #
    #  settings for the size-linewidth prior
    #

    #  use the size-linewidth prior to help inform p_far
    b.prior_velocity_dispersion = False
    #  if `prior_velocity_dispersion = True`, we need to define the beam size
    b.beam = 46*u.arcsec
    #  the following settings inform the size-linewidth relation; the default settings use the values established by Solomon et al. (1987)
    #  power law index for the size-linewidth relation
    b.size_linewidth_index = 0.5
    #  uncertainty in the power law index
    b.size_linewidth_e_index = 0.1
    #  offset for the size-linewidth relation
    b.size_linewidth_sigma_0 = 0.7
    #  uncertainty in the offset
    b.size_linewidth_e_sigma_0 = 0.1

    #  run the BDC; in v2.4 this produces two distance solutions per source
    b.calculate_distances()
    #  choose best distance solution and save final table
    b.get_table_distance_max_probability()


if __name__ == '__main__':
    main()
