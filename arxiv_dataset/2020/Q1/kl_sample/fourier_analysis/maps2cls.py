from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import pymaster as nmt
import flatmaps as fm
from optparse import OptionParser

def opt_callback(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

parser = OptionParser()

parser.add_option('--mask-file',dest='fname_mask',default='NONE',type=str,help='Path to mask file')
parser.add_option('--maps-file',dest='fname_maps',default='NONE',type=str,help='Path to maps file')
parser.add_option('--nbins',dest='nbins',default=7,type=int,help='Number of bins')
parser.add_option('--ell-bins', dest='fname_ellbins', default='NONE', type=str,
                  help='Path to ell-binning file. '+
                  'Format should be: double column (l_min,l_max). One row per bandpower.')
parser.add_option('--output-file', dest='fname_out',default=None,type=str,
                  help='Output file name.')
parser.add_option('--mcm-output', dest='fname_mcm', default='NONE', type=str,
                  help='File containing the mode-coupling matrix. '+
                  'If NONE or non-existing, it will be computed. '+
                  'If not NONE and non-existing, new file will be created')
parser.add_option('--masking-threshold',dest='mask_thr',default=0.5, type=float,
                  help='Will discard all pixel with a masked fraction larger than this.')


####
# Read options
(o, args) = parser.parse_args()

#Read mask
fsk,mskfrac=fm.read_flat_map(o.fname_mask,i_map=0)

#Read bandpowers and create NmtBin
lini,lend=np.loadtxt(o.fname_ellbins,unpack=True)
bpws=nmt.NmtBinFlat(lini,lend)
ell_eff=bpws.get_effective_ells()

#Read maps and create NmtFields
fields=[]
for i in np.arange(o.nbins) :
  fskb,map1=fm.read_flat_map(o.fname_maps,i_map=2*i+0)
  fm.compare_infos(fsk,fskb)
  fskb,map2=fm.read_flat_map(o.fname_maps,i_map=2*i+1)
  fm.compare_infos(fsk,fskb)
  fields.append(nmt.NmtFieldFlat(np.radians(fsk.lx),np.radians(fsk.ly),
                                 mskfrac.reshape([fsk.ny,fsk.nx]),
                                 [map1.reshape([fsk.ny,fsk.nx]),map2.reshape([fsk.ny,fsk.nx])]))

#Read or compute mode-coupling matrix
wsp=nmt.NmtWorkspaceFlat()
if not os.path.isfile(o.fname_mcm) :
  print("Computing mode-coupling matrix")
  wsp.compute_coupling_matrix(fields[0],fields[0],bpws)
  if o.fname_mcm!='NONE' :
    wsp.write_to(o.fname_mcm)
else :
  print("Reading mode-coupling matrix from file")
  wsp.read_from(o.fname_mcm)

#Compute coupled power spectra
cls_coup=[]
ncross=(o.nbins*(o.nbins+1))/2
ordering=np.zeros([ncross,2],dtype=int)
i_x=0
for i in range(o.nbins) :
  for j in range(i,o.nbins) :
    cls_coup.append(nmt.compute_coupled_cell_flat(fields[i],fields[j],bpws))
    ordering[i_x,:]=np.array([i,j])
    i_x+=1
cls_coup=np.array(cls_coup)
#n_cross=len(cls_coup)
#n_ell=len(ell_eff)

#Here we'd do the KL stuff. Instead, right now we just decouple the cls
cls_decoup=np.array([wsp.decouple_cell(c) for c in cl_coup])

#Write output
towrite=[]
towrite.append(ell_eff)
header='[0]-l '
for i_c,c in enumerate(cls_decoup) :
  towrite.append(c)
  header+='[%d]-C(%d,%d) '%(i_c+1,ordering[i_c,0],ordering[i_c,1])
np.savetxt(o.fname_out,np.transpose(np.array(towrite)),header=header)
