import os

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

table_name = 'RD+09_table1_sel'
suffix = ''


def main():
    b = bdw.BayesianDistance()

    #  which version of the BDC to use; set to '1.0' to use BDC v1
    b.version = '2.4'

    #  specify number of CPUs used for multiprocessing
    b.use_ncpus = None

    #  specify the column names of the Galactic Longitude, Galactic Latitude and VLSR values in the input table. If there are no column names, this can also be specified in terms of the column number, i.e. colnr_lon = 0, colnr_lat = 1, etc.
    b.colname_lon = 'GLON'
    b.colname_lat = 'GLAT'
    b.colname_vel = 'Vlsr'
    #  use literature distance solutions to inform the prior for the kinematic distance ambiguity
    b.check_for_kda_solutions = True
    #  use all literature distance solutions in the KDA_info directory apart from the Roman-Duval+09 table
    b.exclude_kda_info_tables = ['Roman-Duval+09']

    #  set weight for spiral arm prior (default: 0.85)
    b.prob_sa = 0.85
    #  set weight for maser parallax prior (default: 0.15)
    b.prob_ps = 0.15
    #  set weight for Galactic Latitude prior (default: 0.85)
    b.prob_gl = 0.85
    #  set weight for kinematic distance prior (default: 0.85)
    b.prob_kd = 0.85

    b.path_to_input_table = os.path.join(
        dirpath_inp,
        '{}.dat'.format(table_name))
    b.path_to_output_table = os.path.join(
        dirpath_out,
        '{}_distance_results{}.dat'.format(table_name, suffix))
    #  specify the table format
    b.table_format = 'ascii'

    #  run the BDC; in v2.4 this produces two distance solutions per source
    b.calculate_distances()
    #  choose best distance solution and save final table
    b.get_table_distance_max_probability()


if __name__ == "__main__":
    main()
