from function_utils import *
import mat73
import json
import os
import argparse
import shutil
import termplotlib as tpl


eng = matlab.engine.start_matlab()
s = eng.genpath('./Matlab_functions')
eng.addpath(s, nargout=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for MRF simulations")

    parser.add_argument("-json", "--json_path", type=str,
                        help="Path_to_config_json_file")

    parser.add_argument("-input_seq", "--input_seq", type=str,
                        help="Path_to_input_seq_param_json")

    args = parser.parse_args()

################### CONFIG FILE ###################
simus_infos = json.load(open(args.json_path))
print(simus_infos)
input_seq = simus_infos['input_seq']

print("Loading external json sequence parameters file")
seqparam = json.load(open(simus_infos['input_seq_path']))
n_echoes = seqparam["n_echoes"] # number of echoes
n_pulses = seqparam["n_pulses"] # number of pulses
N = seqparam["N"] # total length (n_echoes * n_pulses)
FA = seqparam["FA"] # Flip Angle if time-constant FA
FA_train = np.array(seqparam["FA_train"]) # flip angle train
TR_train = np.array(seqparam["TR_train"])* 1e3 # repetition time train
TE_train = np.array(seqparam["TE_train"]) * 1e3 # echo time train
TReff = seqparam["TReff"] # TR effective if multi-echoes
phase = seqparam["phase"] # phase type (constant, cycling,...)
phase_incr = seqparam["phase_incr"] # phase cycling increment if cycling phase type
if "phi_train" in seqparam:
    phi_train = np.array(seqparam["phi_train"]) # final phase train
spoil = seqparam["spoil"] # spoiling "FISP" or "bSSFP" for spoiled or balanced GRE
inv_pulse = seqparam["invPulse"] # inversion pulse True or False
inv_time = simus_infos['inv_time']  # inversion pulse duration in ms
name_seq = simus_infos['name_seq']
T1range = eval(simus_infos['T1range']) # ms
T2range = eval(simus_infos['T2range']) # ms
B1range = eval(simus_infos['B1range'])

commentaries = simus_infos['commentaries']
output_path = simus_infos['output_path']
param = simus_infos['param']
vasc = simus_infos['vasc']
grid = simus_infos['grid']
distrib_path = simus_infos['distrib_path']


############################# DICO PARAMETERS #############################

directory_save_path = output_path + \
                      "Dico" + name_seq + phase

if not os.path.exists(directory_save_path):
    # Create the folder if it doesn't exist
    os.makedirs(directory_save_path)
    print(f"Folder '{directory_save_path}' created successfully.")
else:
    print(f"Folder '{directory_save_path}' already exists.")

json_filename = os.path.basename(args.json_path)
destination_file_path = os.path.join(directory_save_path + '/', json_filename)
shutil.copyfile(args.json_path, destination_file_path)


T1values = np.linspace(T1range[0], T1range[1], T1range[2]) * 1e-3  # sec
T2values = np.linspace(T2range[0], T2range[1], T2range[2]) * 1e-3  # sec
B1values = np.linspace(B1range[0], B1range[1], B1range[2])
print("Loading distribution from 3D vessels geometries...")
distrib = loadmat(distrib_path)
big_structure = distrib['bigStructure']
fn = list(big_structure.keys())
dfvalues = big_structure[fn[0]]['histo']['values']


n_distrib=135000

ndf = dfvalues.shape[0]
nB1 = B1values.shape[0]
nT1 = T1values.shape[0]
nT2 = T2values.shape[0]

print( "Generation of {} signals first, expanded vascular parameters into {} signals".format(
        ndf * nT1 * nT2 * nB1,
        ndf * nT1 * nT2 * nB1 *
        n_distrib))
      
if not os.path.exists(directory_save_path):
    os.makedirs(directory_save_path)
    print('Output directory created at:\n', directory_save_path)
else:
    print("File already exists and will be overwritted at:\n", directory_save_path)

############################## SIMULATIONS #################################
tic = time.time()

# First simulate a Bloch based dictionary with T1,T2,df,B1 if needed
Data_based, Mag_based = dico_based(T1values, T2values, B1values, dfvalues, FA_train, TR_train, TE_train,
                                   phi_train, n_echoes, spoil, vasc, inv_pulse, inv_time, grid, eng)
print('Based dico generated :', Data_based['parameters'].shape[0], 'signals')

# Second expand the basd dictionary using the frequency distributions
print('Generation of expanded dico of', Data_based['parameters'].shape[0] * n_distrib, 'signals', )
Data, Mag = generate_vasc_dico_par_2(Data_based, Mag_based, distrib)

print('Final number of signal after df0 values truncature :{}'.format(Data['parameters'].shape[0]))

toc = time.time()

################################ LOGS IN TEXT FILE ################################
txt_path = directory_save_path + "/Dico.txt"
fd = open(txt_path, 'w')

print('Based dico generated :', Data_based['parameters'].shape[0], 'signals', file=fd)
print('Final number of signal after df0 values truncature :{}'.format(Data['parameters'].shape[0]), file=fd)
print('Ranges:', file=fd)
print('nT1=', nT1, 'nT2=', nT2, 'nB1=', nB1, 'ndf=', ndf, file=fd)
print('T1 from', np.min(Data['parameters'][:, 0]), 'to', np.max(Data['parameters'][:, 0]), 's', file=fd)
print('T2 from', np.min(Data['parameters'][:, 1]), 'to', np.max(Data['parameters'][:, 1]), 's', file=fd)
print('df from', np.min(Data['parameters'][:, 2]), 'to', np.max(Data['parameters'][:, 2]), 'Hz', file=fd)
print('B1 from', np.min(Data['parameters'][:, 4]), 'to', np.max(Data['parameters'][:, 4]), 'Hz', file=fd)

print('Ranges:')
print('nT1=', nT1, 'nT2=', nT2, 'nB1=', nB1, 'ndf=', ndf)
print('T1 from', np.min(Data['parameters'][:, 0]), 'to', np.max(Data['parameters'][:, 0]), 's')
print('T2 from', np.min(Data['parameters'][:, 1]), 'to', np.max(Data['parameters'][:, 1]), 's')
print('df from', np.min(Data['parameters'][:, 2]), 'to', np.max(Data['parameters'][:, 2]), 'Hz')
print('B1 from', np.min(Data['parameters'][:, 4]), 'to', np.max(Data['parameters'][:, 4]), 'Hz')
print('Commentaries:', commentaries, file=fd)
Dico = {'Parameters': Data['parameters'], 'Sequence': Data['sequence'], 'VoxelList': Data['voxellist'], 'MRSignals': Mag}
print('Generation time:', toc - tic, 'seconds', file=fd)
print('Generation time:', toc - tic, 'seconds')

fd.close()

# ################################ SAVING IN .h5 AND .mat ################################
print('Saving in hdf5 format ...')
fname = directory_save_path + "/DICO.h5"
#if os.path.exists(fname):
#    os.remove(fname)
f = h5py.File(fname, 'w')
for data_name in Dico:
    f.create_dataset(data_name, data=Dico[data_name])
f.close()
#
print('Converting .hdf5 file in .mat ...')
mat_path = directory_save_path + "/DICO.mat"
if vasc:
    eng.function_make_dico_struct_vasc(fname, mat_path, nargout=0)
else:
    eng.function_make_dico_struct(fname, mat_path, nargout=0)
eng.quit()

