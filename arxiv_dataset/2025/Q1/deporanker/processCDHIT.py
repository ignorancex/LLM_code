#conda install -c conda-forge biopython
#conda install -c bioconda cd-hit
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def processCDHIT(L,cthresh=0.8,ofile="out.cdhit"):
    """   
    Generate CD-HIT clustering
    Runs CD-HIT and creates a temporary file which is not deleted automatically
    Parameters
    ----------
    L : TYPE Fasta file string OR List of protein sequences OR  SeqRecord
        DESCRIPTION.
    cthresh : TYPE, optional
        DESCRIPTION. Cutoff threshold The default is 0.8.
    ofile : TYPE, optional
        DESCRIPTION. The default is "out.cdhit".

    Returns
    -------
    cc : TYPE Dictionary with cluster id string as key and list of protein ids in each cluster
        DESCRIPTION. 

    """
    if type(L)==type(""):
        ifile = L
    else:
        if type(L[0]==type("")):
            L = [SeqRecord(Seq(p),id=str(i)) for i,p in enumerate(L)]
        ifile = ofile+"_temp.fasta"
        with open(ifile, "w") as output_handle:
            SeqIO.write(L, output_handle, "fasta") 
    
# =============================================================================
#     cmd = "cd-hit -i "+ifile+" -d 0 -o "+ofile+" -c "+str(cthresh)+" -n 3  -G 1 -g 1 -b 20 -l 10 -s 0.0 -aL 0.0 -aS 0.0 -T 4 -M 32000"   
#     os.system(cmd)
# =============================================================================
    with open(ofile+".clstr","r") as fh:
        clusters = fh.readlines()
    cc = {}
    for x in clusters:
        xs = x.split()
        if xs[0]=='>Cluster':
            ccid = xs[1]
            cc[ccid]=[]
        else:
            pid = xs[2][1:].split('...')[0]
            cc[ccid].append(pid)            
    return cc
