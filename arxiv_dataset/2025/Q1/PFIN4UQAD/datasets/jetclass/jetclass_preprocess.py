import uproot
import awkward
import h5py
import numpy as np
import torch
import os
import sys
import json

try:
    file_loc = "raw/" + sys.argv[1] ## should be "train, val, or test"
except:
    file_loc = "raw/val"

all_files = [file_loc + '/' + f for f in os.listdir(file_loc) if f.endswith(".root")]
print("Number of files to process: ", len(all_files))

cats = {}
for f in all_files:
    catname = f.strip().split('/')[-1].split('_')[0]
    if catname not in cats.keys():
        cats[catname] = []
    cats[catname].append(f)

N = len(all_files) // 10

for cat in cats.keys():
    if len(cats[cat]) != N:
        print("file number mismatch for ", cat)
        sys.exit(1)

Np = 60


for ii in range(N):
    Ndata_total = 0
    for jj, cat in enumerate(cats.keys()):
        # print("File: ", cats[cat][ii])
        tree = uproot.open(cats[cat][ii])['tree']
        table = tree.arrays()
        Ndata = len(table['label_QCD'])
        #print("Data Count: ", Ndata)
        Ndata_total += Ndata
        PaddingZ = awkward.Array(np.zeros((Ndata, Np), dtype=float))
        PX = awkward.concatenate((table['part_px'], PaddingZ), axis=1)[:, :Np].to_numpy() ## (Ndata, Np)
        PY = awkward.concatenate((table['part_py'], PaddingZ), axis=1)[:, :Np].to_numpy()
        PT = (PX**2 + PY**2)**0.5
        PTSUM = PT.sum(1).reshape(-1,1)
        PT = PT / PTSUM
        PT = PT.reshape(-1, Np, 1)
        DETA = awkward.concatenate((table['part_deta'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        DPHI = awkward.concatenate((table['part_dphi'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        
        CHARGE = awkward.concatenate((table['part_charge'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        C_HADRON = awkward.concatenate((table['part_isChargedHadron'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        N_HADRON = awkward.concatenate((table['part_isNeutralHadron'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        PHOTON = awkward.concatenate((table['part_isPhoton'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        ELECTRON = awkward.concatenate((table['part_isElectron'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        MUON = awkward.concatenate((table['part_isMuon'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)

        D0VAL = awkward.concatenate((table['part_d0val'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1) / 1000.
        # D0ERR = awkward.concatenate((table['part_d0err'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)
        DzVAL = awkward.concatenate((table['part_dzval'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1) / 1000. 
        # DzERR = awkward.concatenate((table['part_dzerr'], PaddingZ), axis=1)[:, :Np].to_numpy().reshape(-1, Np, 1)

        this_Data = np.concatenate((PT,DETA,DPHI,CHARGE,D0VAL,DzVAL,C_HADRON,N_HADRON,PHOTON,ELECTRON,MUON),-1)
        this_Mask = np.transpose((PT > 0.).astype(float), axes=(0,2,1))

        # Now make augmented data to have the quantities: jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst

        JET_E = table['jet_energy'].to_numpy().reshape(-1,1)
        JET_M = table['jet_sdmass'].to_numpy().reshape(-1,1)
        JET_PT = table['jet_pt'].to_numpy().reshape(-1,1)
        JET_ETA = table['jet_eta'].to_numpy().reshape(-1,1)
        JET_PHI = table['jet_phi'].to_numpy().reshape(-1,1)
        JET_PTSUM  = PTSUM
        #JET_NCONST = table['jet_nparticles'].to_numpy().reshape(-1,1)
        JET_NCONST = this_Mask.sum(-1).reshape(-1,1)

        this_AugData = np.concatenate((JET_E, JET_M, JET_PT, JET_ETA, JET_PHI, JET_PTSUM, JET_NCONST), 1)

        

        # Now make the labels
        
        l_QCD = table['label_QCD'].to_numpy().astype(float).reshape(-1,1)
        l_Hbb = table['label_Hbb'].to_numpy().astype(float).reshape(-1,1)
        l_Hcc = table['label_Hcc'].to_numpy().astype(float).reshape(-1,1)
        l_Hgg = table['label_Hgg'].to_numpy().astype(float).reshape(-1,1)
        
        l_H4q = table['label_H4q'].to_numpy().astype(float).reshape(-1,1)
        l_Hqql = table['label_Hqql'].to_numpy().astype(float).reshape(-1,1)
        l_Zqq = table['label_Zqq'].to_numpy().astype(float).reshape(-1,1)
        l_Wqq = table['label_Wqq'].to_numpy().astype(float).reshape(-1,1)

        l_Tbqq = table['label_Tbqq'].to_numpy().astype(float).reshape(-1,1)
        l_Tbl = table['label_Tbl'].to_numpy().astype(float).reshape(-1,1)

        this_Labels = np.concatenate((l_QCD, l_Hbb, l_Hcc, l_Hgg, l_H4q, l_Hqql, l_Zqq, l_Wqq, l_Tbqq, l_Tbl), 1)

        if jj == 0:
            Data = this_Data
            Mask = this_Mask
            Labels = this_Labels
            AugData = this_AugData
        else:
            Data = np.concatenate((Data, this_Data), 0)
            Mask = np.concatenate((Mask, this_Mask), 0)
            Labels = np.concatenate((Labels, this_Labels), 0)
            AugData = np.concatenate((AugData, this_AugData), 0)
            
        
    if Labels.sum() != Ndata_total:
        print("Sum of labels {} don't match data count {}".format(Labels.sum(), Ndata_total))
        sys.exit(1)

    shuffle_index = np.arange(Ndata_total).astype(int)
    np.random.shuffle(shuffle_index)
    Data = Data[shuffle_index, :, :]
    Mask = Mask[shuffle_index, :, :]
    AugData = AugData[shuffle_index, :]
    Labels = Labels[shuffle_index, :]
        
    # Print file info

    print("File index: ", ii)
    print("Data shape:", Data.shape)
    print("Mask shape:", Mask.shape)
    print("Aug Data shape:", AugData.shape)
    print("Labels shape:", Labels.shape)
    print("Label counts:", Labels.sum(0))

    h5fname = "processed/" + file_loc.split('/')[-1] + '_' + '{}.h5'.format(ii)
    
    test = h5py.File(h5fname, 'w')
    test.create_dataset('particles', data = Data)
    test.create_dataset('masks', data = Mask)
    test.create_dataset('labels', data = Labels)
    test.create_dataset('aug_data', data = AugData)

    test.close()
