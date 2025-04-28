# this notebook has useful functions for bump hunting

import numpy as np
from scipy.special import erfinv
import random

nbkgmax = 13
nsigmax = 8

# apply the cleaning algorithm with threshold thresh
def Cij_algo(Cij, thresh):
    Cij = np.copy(Cij)
    nobs = Cij.shape[0]
    onesm = np.ones([nobs, nobs])
    masks = []
    for iobs in range(nobs):
        mask = np.copy(onesm)
        mask[:,iobs]=0
        mask[iobs,:]=0
        masks.append(mask)
    for _ in range(nobs): # no point cleaning Cij more than nobs times
        scores = (np.sum(Cij,axis=0))/(np.sum(Cij!=0,axis=0))
        scores[np.isnan(scores)]=np.inf
        jr = np.argmin(scores,axis=0)
        maskstot = np.array(masks)[jr]
        maskstot[np.choose(jr, scores)>thresh]=onesm
        maskstot = np.transpose(maskstot)
        Cij= Cij*maskstot
        #else:
        #    continue
    return Cij

# extract counts from Cij files given nbkg background events, nsig signal events and thresh for the algorithm
def extract_counts(CijSS, CijBS, CijBB, nbkg, nsig, thresh):

    nobs = nbkg+nsig

    if nobs<2:
        return np.array([nobs, 10000]).reshape(-1,1)

    if nobs>1:
        Cij3d = np.zeros([nobs, nobs, 10000])
        for i in range (nobs):
            for j in range(i+1, nobs):
                if ((i < nsig) and (j<nsig)):
                    Cij = CijSS
                    Cij3d[i, j, :] = Cij[(Cij[:,0]==i)*(Cij[:,1]==j)][:,2:]
                    Cij3d[j, i, :] = Cij3d[i, j, :]
                elif ((i < nsig) and (j>=nsig)):
                    Cij = CijBS
                    Cij3d[i, j, :] = Cij[(Cij[:,0]==j-nsig)*(Cij[:,1]==i)][:,2:]
                    Cij3d[j, i, :] = Cij3d[i, j, :]   
                elif ((i >= nsig) and (j>=nsig)):
                    Cij = CijBB
                    Cij3d[i, j, :] = Cij[(Cij[:,0]==i-nsig)*(Cij[:,1]==j-nsig)][:,2:]
                    Cij3d[j, i, :] = Cij3d[i, j, :]
        Cij_clean = Cij_algo(Cij3d,thresh)
        scores = (np.sum(Cij_clean,axis=0))/(np.sum(Cij_clean!=0,axis=0))
        scores[np.isnan(scores)]=0
        counts = []
        for i in range(nobs+1): # to avoid missing stuff, like no bin with all nobs events inside
            counts.append([i, np.sum(np.sum(scores>0, axis=0)==i)] )
        counts[1][1]=counts[0][1] # just switch 0 and 1 for consistency with bumphunt
        counts = np.array(counts).T[:,1:]
        return counts

# extract TS from Cij files given a poisson sample with pairs of (nsig,nbkg) pairs with their probability and thresh for the algorithm
def extract_TS(CijSS, CijBS, CijBB, pois_sample, thresh):
    un_pairs, un_counts =  np.unique(pois_sample,axis=1, return_counts=True)
    # discard when more events than bkg and signal events considered
    index_good = (un_pairs[0]<=nbkgmax)*(un_pairs[1]<=nsigmax) 
    un_pairs, un_counts = un_pairs[:,index_good], un_counts[index_good]

    TS_dist = []
    for _ in range(len(un_counts)):
        nbkg = int(un_pairs[0,_])
        nsig = int(un_pairs[1,_])
        nobs = nbkg+nsig

        if nobs<2:
            TS_dist.append(nobs*np.ones(un_counts[_]))

        if nobs>1:
            counts = extract_counts(CijSS, CijBS, CijBB, nbkg, nsig, thresh)
            TS_dist.append(np.array(random.choices(counts[0], weights=counts[1], k =un_counts[_])))
    TS_dist =  np.array(np.hstack((TS_dist)))
    return np.array(np.unique(TS_dist.astype(int), return_counts=True))

# this extract the diphoton invariant mass from the decuplet of variables that identify a diphoton event
def mgg(X):
    nobs = int(X.shape[1]/10)
    mjj_arr =[]
    for i in range(nobs):
        Xfeat = X[:, int(10*i):int(10*(i+1))]
        E1, E2 = np.exp(Xfeat[:,0]), np.exp(Xfeat[:,1])
        p1 = np.exp(Xfeat[:,0])*np.vstack(([np.sin(np.exp(Xfeat[:,6]))*np.cos(Xfeat[:,8]),np.sin(np.exp(Xfeat[:,6]))*np.sin(Xfeat[:,8]),np.cos(np.exp(Xfeat[:,6]))]))
        p2 = np.exp(Xfeat[:,1])*np.vstack(([np.sin(np.exp(Xfeat[:,7]))*np.cos(Xfeat[:,9]),np.sin(np.exp(Xfeat[:,7]))*np.sin(Xfeat[:,9]),np.cos(np.exp(Xfeat[:,7]))]))
        mjj_arr.append(np.sqrt((E1+E2)**2-np.sum((p1+p2)**2,axis=0)))
    return np.array(mjj_arr)

# perform bumphunting with separated bins (sep) or overlapping bins (over)
class bphunt():
    def __init__(self, mjj, nbins, bin_range):
        self.nobs = mjj.shape[0]
        self.nbins = nbins
        bins = np.linspace(bin_range[0], bin_range[1], self.nbins)
        binid = np.digitize(mjj, bins)
        self.bin_count = np.array([np.sum((binid==i), axis=0) for i in range(nbins+1)]) # including underflow and overflow

    def counting(self):
        return self.bin_count
        #counting is always performed anyway, but not necessarily returned
    
    def sep(self):
        counts =[]
        for i in range(1, self.nobs+1): # to avoid missing stuff, like no bin with all nobs events inside
            counts.append([i, np.sum(np.max(self.bin_count, axis=0)==i)] )
        return np.array(counts).T

    # with bin_over = 1 it defaults to bphunt, but throws away underflow and overflow bins
    def over(self, bin_over):
        counts = []
        bin_count_sum =  np.zeros([self.nbins-bin_over, self.bin_count.shape[1]], int)
        for i in range(bin_over):
            bin_count_sum += self.bin_count[1+i:self.nbins-bin_over+i+1]
        for i in range(1, self.nobs+1): # to avoid missing stuff, like no bin with all nobs events inside
            counts.append([i, np.sum(np.max(bin_count_sum, axis=0)==i)] ) # ignore underflow and overflow (?)
        return np.array(counts).T

# extract performance values from TS (p-value, log p-value, Z sensitivity)
class perf_pval():
    def __init__(self, countsB, countsS):
        self.countsB = countsB
        self.countsS = countsS

        nobs = np.max([countsB[0].max(),countsS[0].max()])
        
        pvals = []
        probs = []
        for i in range(nobs+1):
            pvals.append(np.sum(self.countsB[1][self.countsB[0]>=i])/np.sum(self.countsB[1]))
            probs.append(np.mean(self.countsS[1][self.countsS[0]==i])/np.sum(self.countsS[1]))
        self.pvals, self.probs = np.array(pvals), np.array(probs)
        self.probs[np.isnan(self.probs)]=0
    def meanp(self):
        return np.sum(self.pvals*self.probs)
    def meanlogp(self):
        pvals2 = np.copy(self.pvals) # do not overwrite it when regularizing
        pvals2[self.pvals == 0 ]= 1/np.sum(self.countsB[1]) # regularize for log
        return np.sum(-np.log(pvals2)*self.probs)
    def meansigma(self):
        pvals2 = np.copy(self.pvals) # do not overwrite it when regularizing
        pvals2[self.pvals == 0 ]=1/np.sum(self.countsB[1]) # regularize for log
        return np.sum(np.sqrt(2)*erfinv(1-pvals2)*self.probs)

# extract counts from an imported dataframe    
def count_from_df(df, nnurbkg, nbkgsig):
    countsB_temp, countsS_temp = [], []
    for ibkg in range(nbkgmax+1):
        for isig in range(nsigmax+1):
            temp = list(df.loc[(df["nbkg"]==ibkg) * (df["nsig"]==isig)]["counts"])[0]
            cb = np.sum((nnurbkg[0]==ibkg)*(nnurbkg[1]==isig))
            cs = np.sum((nbkgsig[0]==ibkg)*(nbkgsig[1]==isig))
            if cb:
                countsB_temp.append(random.choices(temp[0], weights=temp[1], k =cb))
            if cs:
                countsS_temp.append(random.choices(temp[0], weights=temp[1], k =cs))

    countsS = np.array([x for xs in countsS_temp     for x in xs ])
    countsB = np.array([x for xs in countsB_temp     for x in xs ])

    countB, countS = [], []
    for i in range(nbkgmax+nsigmax+1): # to avoid missing stuff, like no bin with all nobs events inside
        countB.append([i, np.sum(countsB==i)] )
        countS.append([i, np.sum(countsS==i)] )
    return np.array(countB).T, np.array(countS).T

# extract scores from Cij after cleaning algorithm
def extract_scores(CijSS, CijBS, CijBB, nbkg, nsig, thresh):

    nobs = nbkg+nsig

    if nobs>1:
        Cij3d = np.zeros([nobs, nobs, 10000])
        for i in range (nobs): # ok here not to have +1
            for j in range(i+1, nobs):
                if ((i < nsig) and (j<nsig)):
                    Cij = CijSS
                    Cij3d[i, j, :] = Cij[(Cij[:,0]==i)*(Cij[:,1]==j)][:,2:]
                    Cij3d[j, i, :] = Cij3d[i, j, :]
                elif ((i < nsig) and (j>=nsig)):
                    Cij = CijBS
                    Cij3d[i, j, :] = Cij[(Cij[:,0]==j-nsig)*(Cij[:,1]==i)][:,2:]
                    Cij3d[j, i, :] = Cij3d[i, j, :]   
                elif ((i >= nsig) and (j>=nsig)):
                    Cij = CijBB
                    Cij3d[i, j, :] = Cij[(Cij[:,0]==i-nsig)*(Cij[:,1]==j-nsig)][:,2:]
                    Cij3d[j, i, :] = Cij3d[i, j, :]
        Cij_clean = Cij_algo(Cij3d,thresh)
        scores = (np.sum(Cij_clean,axis=0))/(np.sum(Cij_clean!=0,axis=0))
        scores[np.isnan(scores)]=0
        return scores