import numpy as np
clinical_file = 'F:\cancer_data\clinical_sample_data.txt'
mRNA_file = 'F:\cancer_data\data_mRNA_seq_fpkm_polya_56M.txt'

a = np.loadtxt(clinical_file)
print(a)