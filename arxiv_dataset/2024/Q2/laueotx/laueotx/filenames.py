import os

def get_filepath_results(dir_out, tag):

	return os.path.join(dir_out, f'grainspotter_latest_state_{tag}.pkl')

def get_filepath_tfrecords(dir_out, index, tag='training'):

	return os.path.join(dir_out, f'spots_{tag}_{index:04d}.tfrecords')

def get_filepath_morphology_training(dir_out, index, out_format='h5'):

	if out_format == 'h5':
		return os.path.join(dir_out, f'sample_morphology_traninig_{index:03d}.h5')
	elif out_format=='tfrecords':
		return os.path.join(dir_out, f'sample_morphology_traninig_{index:03d}.tfrecords')

def get_filename_merged(dir_out, sig_s, frac_outliers, n_grn_sample, n_trials, opt_fun, merge_partials=False):

    sig_str = str(sig_s).replace('.', 'p')
    tag = f'{opt_fun}_ngain{n_grn_sample}_ntrials{n_trials}_noisesig{sig_str}_outs{int(frac_outliers):d}'
    if merge_partials:
    	tag += '_fracmerge'
    fname_out = os.path.join(dir_out, f'test_analyze_sample_scan__{tag}_merged.h5')
    return fname_out


def get_filename_output(dir_out, sig_s, frac_outliers, n_grn_sample, n_trials, opt_fun, index):

    sig_str = str(sig_s).replace('.', 'p')
    
    # if frac_outliers==0:
    #     tag = f'{opt_fun}_ngain{n_grn_sample}_ntrials{n_trials}_noisesig{sig_str}_part{index}'
    # else:
    tag = f'{opt_fun}_ngain{n_grn_sample}_ntrials{n_trials}_noisesig{sig_str}_outs{int(frac_outliers):d}_part{index}'

    fname_out = os.path.join(dir_out, f'test_analyze_sample_scan__{tag}.h5')
    return fname_out


def get_filename_output_optstats(dir_out, sig_s, frac_outliers, n_grn_sample, n_trials, opt_fun, index):

    sig_str = str(sig_s).replace('.', 'p')
    
    # if frac_outliers==0:
    #     tag = f'{opt_fun}_ngain{n_grn_sample}_ntrials{n_trials}_noisesig{sig_str}_part{index}'
    # else:
    tag = f'{opt_fun}_ngain{n_grn_sample}_ntrials{n_trials}_noisesig{sig_str}_outs{int(frac_outliers):d}_part{index}_optstats'

    fname_out = os.path.join(dir_out, f'test_analyze_sample_scan__{tag}.h5')
    return fname_out


def get_filename_realdata_part(dir_out, tag, index):

    fname_out = os.path.join(dir_out, f'analyze_sample__{tag}__part{index:03d}.h5')
    return fname_out

def get_filename_realdata_merged(dir_out, tag):

    fname_out = os.path.join(dir_out, f'analyze_sample__{tag}__merged.h5')
    return fname_out