# Libraries to load
import numpy,scipy,math
import os,time
import matplotlib
import matplotlib.pyplot as plt
import csv
from src.functions import alpha, Resonance
import src.pars as par
import src.chan as ch
import src.data as data

class Utilities():
    
    def writefiles(self,path,x,y):
        with open(path, 'w') as txtfile:
                for i in range(0,len(x)):
                    txtfile.write("%s \t %s\n" %(x[i],y[i]))
                txtfile.close()
                
    #function that reads a table in a .txt file and converts it to a numpy array
    def readfile(filename):
        array = []
        with open(filename) as f:
            for line in f:
                if line[0]=="#":continue
                words = [float(elt.strip()) for elt in line.split( )]
                array.append(words)
        return numpy.array(array)
    
    def arrdiv(self,arrA,arrB):
        div = []
        for i in range(0,len(arrA)):
            if arrB[i]==0:
                div.append(0)
            elif arrB[i]!=0:
                div.append(arrA[i]/arrB[i])
        return div
    
    def intdiv(self,n,d):
        return n/d if d else 0
    
class Processes():
    ######################
    ### cross sections ###
    ######################
    
    # e+e- -> mu+mu- #
    def xSection2e2mu(self,Q2):
        sigma2e2mu = (4 * math.pi * alpha.alphaEM(Q2)**2 * par.gev2nb)/(3*Q2)
        return sigma2e2mu
    
    # perturbative decay into quarks
    #---------------  eq. (9.7) of PDG - Quantum Chromodynamics  -------------------# 
    
    def RInclusive(self,Q):
        sumQq = self.cMed_u+self.cMed_d+self.cMed_s
        sumQq2 = self.cMed_u**2+self.cMed_d**2+self.cMed_s**2
        nf=3
        if (2*par.mD0)<=Q:
            sumQq += self.cMed_c
            sumQq2 += self.cMed_c**2
            nf += 1
        if (par.mUps)<Q:
            sumQq += self.cMed_b
            sumQq2 += self.cMed_b**2
            nf += 1
        nc = 3
        Rew = (nc * sumQq2)
        eta = sumQq**2/(3*sumQq2)
        coeff = [1.,1.9857 - 0.1152*nf,-6.63694 - 1.20013*nf - 0.00518*nf**2 - 1.240*eta,
             -156.61 + 18.775*nf - 0.7974*nf**2 + 0.0215*nf**3-(17.828 - 0.575*nf)*eta]
        lambQCD =0
        if Q>1.:
            for n in range(0,4):
                lambQCD += coeff[n]*(alpha.alphaQCD(Q)/math.pi)**(n+1)
        return Rew*(1+lambQCD)
   
    
    ####################
    ### decay widths ###
    ####################
    
    ### widths with SM final states ###
    # vector -> f fbar
    def GammaVff(self,Cf, g, m, xF, mF):
        if m<2*mF:
            return 0
        else:  
            pre = Cf*(g* xF)**2/12/math.pi
            kin = m*(1 + 2*(mF**2/m**2))*numpy.sqrt(1- 4*(mF**2/m**2))
            return pre*kin
    
    def calcWidthQuarksDP(self,g):
        mass = []
        widthscc = []
        widthsbb = []
        widthstt = []
        m = 0
        while m < 3.0 :
            mass.append(m)
            width2cc = self.GammaVff(3.,g,m,2*par.ge/3,par.mc_)
            width2bb = self.GammaVff(3.,g,m,-1*par.ge/3,par.mb_)
            width2tt = self.GammaVfff(3.,g,m,2*par.ge/3,par.mt_)
            widthscc.append(width2cc)
            widthsbb.append(width2bb)
            widthstt.append(width2tt)
            m+=0.001
        return (mass,widthscc,widthsbb,widthstt)
  
    
    ### Widths with DM final states ###
    
    # vector -> DM DM
    def Gamma2DM(self,g, mV, mDM,DMtype="No", splitting=0.1): # Vector Boson -> X X (DM)
        if mV<2*mDM: return 0
        #model-independent prefactor for two body decay
        pre = g**2/48./math.pi*mV*(1-4*mDM**2/mV**2)**(1/2)/mV**2
        #model-dependent matrix element
        me = 0.
        if DMtype in ["complex scalar","Majorana","Majorana fermion"]:
            me = mV**2*(1- (4*mDM**2/mV**2))
            # identical particle factor
            if DMtype in ["Majorana","Majorana fermion"]: me*=2.
        elif DMtype in ["Dirac","Dirac fermion"]:
            me = 4*mV**2*(1+2*mDM**2/mV**2)
        elif DMtype=="inelastic":
            mDM2 = mDM*(splitting+1)
            if mV<mDM+mDM2: return 0
            else:
                pre = g**2/48./math.pi*((1-(mDM+mDM2)**2/mV**2)*(1-(mDM*splitting)**2/mV**2))**0.5
                me = 4.*mV*((1+2*mDM*mDM2/mV**2)-(mDM*splitting)**2*(1+(mDM+mDM2)**2/mV**2)/2./mV**2) 
        else:
            print("DM type not specified")
        return pre*me
    
    
                
class Model():
    
    def __init__(self,name):
        self.model_name = name
        self.couplings = []
        self.DMtype = "No"
        self.gDM = 1.
        self.gQ = 1.
        self.R = 3.
        self.split = 0.1
        
    def set_charges(self,c):
        if len(c)==12:
            self.couplings = c
        else: print("Coupling list has wrong size")
            
    def set_DMtype(self, DM,Rchi=3,gDM=1.,splitting=0.1):
        if DM not in ch.allDMtypes:
            print("Unclear specification of DM type. Possible candidates are")
            print("complex scalar")
            print("Majorana fermion")
            print("Dirac fermion")
            print("If the vector mediator does not couple to DM, just set 'DMtype='No'") 
        else:
            self.DMtype = DM
            self.R = Rchi
            self.gDM = gDM
            if DM=="inelastic":
                self.split = splitting
        
    def set_folders(self):
        # create model folder 
        if isinstance(self.model_name, str) and self.DMtype != None:
            folderType = 'SM'
            if self.DMtype !="No":
                folderType = 'DM_'+self.DMtype.replace(" ", "_")
            if not os.path.exists("models/"):
                os.makedirs("models/")
            if os.path.isdir("models/"+self.model_name.replace(" ", "_")+"_"+folderType):
                print("model ", self.model_name," with ",folderType, " couplings already exists!")
                print("If you choose to save files for this model in the following, existing files will be overwritten.")
            else:
                os.mkdir("models/"+self.model_name.replace(" ", "_")+"_"+folderType)
                os.mkdir("models/"+self.model_name.replace(" ", "_")+"_"+folderType+"/widths")
                os.mkdir("models/"+self.model_name.replace(" ", "_")+"_"+folderType+"/brs")
                os.mkdir("models/"+self.model_name.replace(" ", "_")+"_"+folderType+"/Rvalues")
                os.mkdir("models/"+self.model_name.replace(" ", "_")+"_"+folderType+"/plots")
                print("new model called ", self.model_name," with ",folderType," couplings")
        else:
            print("First initialize the model and set a DM type with set_DMtype()")


    
class Widths(Utilities,Processes):
    
    def __init__(self,model):
        self.modelname = model.model_name
        self.DM = model.DMtype
        self.cMed_d = model.couplings[0]
        self.cMed_u = model.couplings[1]
        self.cMed_s = model.couplings[2]
        self.cMed_c = model.couplings[3]
        self.cMed_b = model.couplings[4]
        self.cMed_t = model.couplings[5]
        self.clep = [model.couplings[i] for i in range(6,len(model.couplings))]
        self.Rval = model.R
        self.gDM = model.gDM
        self.gQ = model.gQ
        self.split = model.split
        self.masses = []
        self.wtotal=[]
        self.wqcd=[]
        self.whad,self.wquark,self.wlep,self.wDM=[],[],[],[]
        self.widthshad = {}
        self.widthslep = {}
        self.widthspert = {}
        self.singlehad = {}
        self.mhad= None
    
    # Decay Width
    def calc_part(self,mmax=2.0):
        ## reset dark param
        for ilabel in ch.labelhad:
            self.widthshad[ilabel] = []
        for ilabel in ch.labellep:
            self.widthslep[ilabel] = []
        self.widthspert["quarks"] = []
        if self.DM != "No": self.widthspert["DM"] = []
        self.masses = []
        self.whad,self.wquark,self.wlep,self.wDM=[],[],[],[]
        # initialization
        for channel in ch.Fchannels:
            channel.resetParameters(self.gQ,0.,0.,0.,self.cMed_u,self.cMed_d,self.cMed_s)
        # calc Widths
        m = 2* par.me_ 
        start_time = time.time()
        imass=0
        nmassval = round((mmax-m)/0.001)
        while m < mmax:
            self.masses.append(m)
            wHtot,wLtot,wDMtot = 0,0,0
            for i in range(0, 17):
                width = 0.
                if m < 2.0: width = getattr(ch.Fchannels[i], 'GammaDM')(m)
                width *= self.gQ**2
                self.widthshad[ch.labelhad[i]].append(width)
                wHtot += width
            w_ppbar,w_nnbar = 0.,0.
            if m<2.0:
                w_ppbar = ch.Fppbar.GammaPDM(m)
                w_ppbar *= self.gQ**2
                w_nnbar = ch.Fppbar.GammaNDM(m)
                w_nnbar *= self.gQ**2
            self.widthshad["ppbar"].append(w_ppbar)
            self.widthshad["nnbar"].append(w_nnbar)
            # sum hadrons
            wHtot += w_ppbar+w_nnbar
            self.whad.append(wHtot) 
            # leptons
            coeff = [1.,1.,1.,1/2,1./2,1./2]
            for i in range(0,len(par.mlep_)):
                wlep = self.GammaVff(coeff[i],self.gQ,m,self.clep[i],par.mlep_[i])
                self.widthslep[ch.labellep[i]].append(wlep)
                wLtot+=wlep
            self.wlep.append(wLtot) 
            # quarks
            norm = self.GammaVff(1,self.gQ,m,1,par.mlep_[1])
            qwidth =self.RInclusive(m)*norm
            self.widthspert["quarks"].append(qwidth)
            self.wquark.append(qwidth)
            # DM
            if self.DM !="No":
                if self.DM=="inelastic": width2DM = self.Gamma2DM(self.gDM, m, m/self.Rval,self.DM,splitting=self.split)
                else: width2DM = self.Gamma2DM(self.gDM, m, m/self.Rval,self.DM)
                self.widthspert["DM"].append(width2DM)
                wDMtot += width2DM
            self.wDM.append(wDMtot)     
            #print (mDM)
            imass+=1
            if (imass+1)%100 == 0: 
                print ("processed up to m=",round(m,2),"GeV (", imass+1, "/", nmassval+1, "mass values) in", round(time.time()-start_time,3), "seconds")
            #m+=0.01
            m+=0.001

    def had2quark(self,mhad=None):
        print ("Calculating the hadron-quark transition...")
        if mhad != None:
            if mhad<1.5:
                print ('Please, choose a transition value above 1.5 GeV.')
            else:
                self.mhad = mhad
        else:
            intx = numpy.argwhere(numpy.diff(numpy.sign(numpy.asarray(self.wquark) - numpy.asarray(self.whad)))).flatten()
            if intx.size >0 and self.masses[intx[-1]]>=1.5:
                self.mhad = self.masses[intx[-1]]
                print ("The transition from hadrons to quarks will happen at "+str(round(self.mhad,3))+" GeV.")
            elif self.masses[intx[-1]]<1.5:
                print ("The value of mass found for the intersection mhad=",round(self.masses[intx[-1]],3),"is inappropriate (below 1.5 GeV). Please use calc(mhad= <transition_value_GeV>) to set by hand the transition mass.")
            else: 
                print ("The function failed to find a intersection mass between the hadronic and the perturbative quark width. Please use calc(mhad= <transition_value_GeV>) to set by hand the transition mass.")
    
    def calc_total(self,mmax=2.0):
        self.wtotal=[]
        self.wqcd=[]
        print ("Calculating the total width...")
        if self.mhad==0 or self.mhad==None:
            print ("Failed to find the hadron-quark mass transition value for this model. Please, insert mhad by hand.")
            return
        for i in range(0, len(self.masses)):
            wTot=0
            if self.masses[i]<= self.mhad:
                self.wqcd.append(self.whad[i])
                wTot+= self.whad[i]+self.wlep[i]+self.wDM[i]
            if self.masses[i]> self.mhad:  
                self.wqcd.append(self.wquark[i])
                wTot+= self.wquark[i]+self.wlep[i]+self.wDM[i]
            self.wtotal.append(wTot)
        print ("Done.")
        
    def calc(self,mmax=2.0,mhad=None):
        self.calc_part(mmax)
        self.had2quark(mhad)
        if self.mhad==None or self.mhad<=1.5:
            return "intersection mass between the hadronic and the perturbative quark width has to be chosen first"
        self.calc_total(mmax)

                        
    def save(self):
        Wtype= None
        if self.DM != "No":
            Wtype = 'DM_'+self.DM.replace(" ", "_")
        elif self.DM=="No":
            Wtype = 'SM'
        widths = {**self.widthshad,**self.widthslep,**self.widthspert}
        mstr = self.modelname.replace(" ","_")
        for channel in ch.label:
            # SM widths
            self.writefiles('models/'+mstr+"_"+Wtype+'/widths/width_{0}.txt'.format(channel),x=self.masses,y=widths[channel])
        # DM widths
        if self.DM != "No":
            self.writefiles('models/'+mstr+"_"+Wtype+'/widths/width_'+Wtype+'.txt', x=self.masses,y=widths["DM"])
        # total width
        self.writefiles('models/'+mstr+"_"+Wtype+'/widths/'+mstr+'_width_total.txt',x=self.masses,y=self.wtotal)
        # total hadronic+quarks width
        self.writefiles('models/'+mstr+"_"+Wtype+'/widths/'+mstr+'_width_qcd.txt',x=self.masses,y=self.wqcd)
        #lifetime
        tau = self.arrdiv(arrA=[par.hbar]*len(self.wtotal),arrB= self.wtotal)        
        self.writefiles('models/'+mstr+"_"+Wtype+'/'+mstr+'_lifetime_tau_sec.txt',x=self.masses,y=tau)
        ctau = self.arrdiv(arrA=[par.hbar*par.cLight]*len(self.wtotal),arrB= self.wtotal)  
        self.writefiles('models/'+mstr+"_"+Wtype+'/'+mstr+'_decay_length_ctau_meters.txt',x=self.masses,y=ctau)
        

    def calc_single(self,channel):
        Wtype= None
        if self.DM != "No":
            Wtype = 'DM_'+self.DM.replace(" ", "_")
        elif self.DM=="No":
            Wtype = 'SM'
        self.singlehad[channel]=[]
        for imass in range(0,len(self.masses)):
            func,mode = ch.allchannels.get(channel)
            if mode<0:
                self.singlehad[channel].append(getattr(func,"GammaDM")(float(self.masses[imass])))
            else:
                self.singlehad[channel].append(getattr(func,"GammaDM_mode")(float(self.masses[imass]),int(mode)))
        self.writefiles('models/'+self.modelname.replace(" ","_")+"_"+Wtype+'/widths/singlewidth_{0}.txt'.format(channel)
                        ,x=self.masses,y=self.singlehad[channel])
    
  
            
    # function to plot Widths
    #---------------  plot -------------------# 
    def plot(self,xrange=[0.1,2.],yrange=[1.e-3,2.], name=None, Wsingle_had=[],Wsingle_lep=[], W_DM=False, Wpert=False):
        fig, ax = fig, ax = plt.subplots(figsize=(9., 4.8))
        ax.plot(self.masses,self.wtotal, c='darkviolet', lw =1.3, label='total width')
        all_had = {**self.widthshad,**self.singlehad}
        if len(Wsingle_had)>0:
            for xw in Wsingle_had:
                if xw=='had': ax.plot(self.masses,self.whad, lw=1.2, label='hadrons')
                else: ax.plot(self.masses,all_had[xw], lw=1.2, label=xw)
        if len(Wsingle_lep)>0:
            for xw in Wsingle_lep:
                if xw == 'lep': ax.plot(self.masses,self.wlep, lw=1.2, label='leptons')
                else: ax.plot(self.masses,self.widthslep[xw], lw=1.2, label=xw)
        if Wpert==True:
            ax.plot(self.masses,self.wquark,lw=1.2,linestyle="dashed",label="quarks")
        ## DM width ##  
        Wtype= None
        if self.DM != "No":
            Wtype = 'DM_'+self.DM.replace(" ", "_")
            if W_DM==True:
                ax.plot(self.masses,self.wDM,lw=1.2,linestyle="dashdot",label= self.DM)
        elif self.DM=="No":
            Wtype = 'SM'            
        ###############
        ax.set_title(self.modelname)
        ax.set_xlabel("$m_{Z_Q}$ [GeV]",fontfamily= 'serif', fontsize=14)
        ax.set_ylabel("$\Gamma_{tot}$",fontfamily= 'serif',fontsize=14)
        ax.legend(ncol=2)      
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        plt.axvline(self.mhad,ls='--',label="transition")
        plt.yscale("log")
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.2)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if name:
            try:
                plt.savefig('models/'+self.modelname.replace(" ","_")+"_"+Wtype+'/plots/'+name)
            except:
                print ("Please choose a valid name+extension, e.g. plot(name='plot_1.pdf').")
        plt.show()
        plt.close()
        return fig

class Branching(Utilities):

    def __init__(self,widths):
        self.modelname = widths.modelname
        self.DM = widths.DM
        self.wsinglehad = widths.singlehad
        self.wtotal = widths.wtotal
        self.wqcd = widths.wqcd
        self.widthshad = widths.widthshad
        self.widthslep = widths.widthslep
        self.widthspert = widths.widthspert
        self.masses = widths.masses
        self.mhad = widths.mhad
        self.BRshad = {}
        self.BRsinglehad = {}
        self.BRslep = {}
        self.BRspert = {}
        self.BRqcd = []
        self.BRvis = []
        self.BRinv = []
 
    def calc(self):
        for ilabel in ch.labelhad:
            self.BRshad[ilabel] = []
        for ilabel in ch.labellep:
            self.BRslep[ilabel] = []
        self.BRspert["quarks"] = []
        if self.DM !="No":
            self.BRspert["DM"] = []
        #hadronic channels
        for ichannel in ch.labelhad:
            brlist = self.arrdiv(arrA=self.widthshad[ichannel],arrB= self.wtotal)
            self.BRshad[ichannel]= brlist
        #leptonic channels
        for ichannel in ch.labellep:
            brlist = self.arrdiv(arrA=self.widthslep[ichannel],arrB= self.wtotal)
            self.BRslep[ichannel] = brlist
        #quarks
        brlist = self.arrdiv(arrA=self.widthspert["quarks"],arrB= self.wtotal)
        self.BRspert["quarks"] = brlist
        #DM channel
        if self.DM !="No":
            brlist = self.arrdiv(arrA=self.widthspert["DM"],arrB= self.wtotal)
            self.BRspert["DM"] = brlist
        # hadrons+quarks 
        self.BRqcd = self.arrdiv(arrA=self.wqcd,arrB= self.wtotal)
        #hadronic
        haddic = self.BRshad.values()
        self.BRhad = [sum(x) for x in zip(*haddic)]
        #leptonic
        lepdic = self.BRslep.values()
        self.BRlep = [sum(x) for x in zip(*lepdic)]
        #invisible and visible
        self.BRvis = [w+x+y+z for w,x,y,z in zip(self.BRqcd,self.BRslep["elec"],self.BRslep["muon"],self.BRslep["tau"])]
        #self.BRvis = self.BRqcd+self.BRslep["elec"]+self.BRslep["muon"]+self.BRslep["tau"]
        self.BRinv = [x+y+z for x,y,z in zip(self.BRslep["nue"],self.BRslep["numu"],self.BRslep["nutau"])]
        #self.BRinv = self.BRslep["nue"]+self.BRslep["numu"]+self.BRslep["nutau"]
        if self.DM !="No":
            self.BRinv = [i+j for i,j in zip(self.BRinv,self.BRspert["DM"])]
                
    def calc_single(self,channel):
        if self.DM !="No":
            brType = 'DM_'+self.DM.replace(" ", "_")
        if self.DM =="No":
            brType = 'SM'
        #single channel
        try:
            brlist = self.arrdiv(arrA=self.wsinglehad[channel],arrB= self.wtotal)
            self.BRsinglehad[channel] = brlist
            #save
            self.writefiles('models/'+self.modelname.replace(" ","_")+"_"+brType+'/brs/singlebfrac_{0}.txt'.format(channel),
                            self.masses,self.BRsinglehad[channel])
        except KeyError:
            print ("First calculate the specific hadronic channel width with widths.calc_single(channel_name)")
        

    # function to plot branching ratios ["red","deepskyblue", "blue","green"]
    #---------------  plot -------------------# 
    def plot(self,xrange=[0.01,2.],yrange=[1.e-3,2.], BRsingle_had=[],BRsingle_lep=[],BRDM=None,name=None):
        Wtype= None
        if self.DM != "No":
            Wtype = 'DM_'+self.DM.replace(" ", "_")
        elif self.DM=="No":
            Wtype = 'SM'
        fig, ax = fig, ax = plt.subplots(figsize=(9., 4.8))
        ax.plot(self.masses,self.BRqcd, c='red', lw =1.3, label='$\\mathcal{F} =$hadrons')
        ax.plot(self.masses,self.BRslep["elec"], c='deepskyblue', lw =1.3, label='$\\mathcal{F} = e^{+}e^{-}$')
        ax.plot(self.masses,self.BRslep["muon"], c='blue', lw =1.3, label='$\\mathcal{F} = \mu^{+}\mu^{-}$')
        if any(self.BRinv):
            ax.plot(self.masses,self.BRinv, c='green', lw =1.3, label='$\\mathcal{F} =$ invisible')
        all_BRhad = {**self.BRshad,**self.BRsinglehad}
        if len(BRsingle_had)>0:
            for xbr in BRsingle_had:
                ax.plot(self.masses,all_BRhad[xbr], lw=1.2, label="$\\mathcal{F} =$"+xbr)
        if len(BRsingle_lep)>0:
            for xbr in BRsingle_lep:
                ax.plot(self.masses,self.BRslep[xbr], lw=1.2, label="$\\mathcal{F} =$"+xbr)
        if BRDM != None: ax.plot(self.masses,self.BRspert["DM"],lw=1.2,label=BRDM)
        ax.set_title(self.modelname)
        ax.set_xlabel("$m_{Z_Q}$ [GeV]",fontsize=14)
        ax.set_ylabel("Br ($\\; Z_{Q} \\; \\to \\; \\mathcal{F} \\;$)",fontfamily= 'sans-serif',fontsize=14)
        ax.legend(ncol=2)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        plt.axvline(self.mhad,ls='--',label="transition")
        plt.yscale("log")
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.2)
        if name:
            try:
                plt.savefig('models/'+self.modelname.replace(" ","_")+"_"+Wtype+'/plots/'+name)
            except:
                print ("Please choose a valid name+extension, e.g. plot(name='plot_1.pdf').")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        plt.close()
        return fig 
    
    def save(self):
        if self.DM !="No":
            brType = 'DM_'+self.DM.replace(" ", "_")
        if self.DM =="No":
            brType = 'SM'
        mstr = self.modelname.replace(" ","_")
        #all hadronic and leptonic Branching Ratios +DM if specified
        all_BRs = {**self.BRshad,**self.BRslep,**self.BRspert}
        for ichannel in ch.label:
            self.writefiles('models/'+mstr+"_"+brType+'/brs/bfrac_{0}.txt'.format(ichannel),
                            self.masses,all_BRs[ichannel])
        # visible and invisible branching ratios
        # visible
        self.writefiles('models/'+mstr+"_"+brType+'/bfrac_visible.txt',self.masses,self.BRvis) 
        # invisible
        self.writefiles('models/'+mstr+"_"+brType+'/bfrac_invisible.txt',self.masses,self.BRinv) 
        #hadrons+quarks
        self.writefiles('models/'+mstr+"_"+brType+'/bfrac_qcd.txt',self.masses,self.BRqcd)
        #leptons
        self.writefiles('models/'+mstr+"_"+brType+'/bfrac_leptons.txt',self.masses,self.BRlep)
        #DM
        if self.DM !="No":
            self.writefiles('models/'+mstr+"_"+brType+'/bfrac_DM.txt',self.masses, self.BRspert["DM"])

            
class Rvalues(Utilities,Processes):
    
    def __init__(self,widths):
        self.modelname = widths.modelname
        self.DM = widths.DM
        self.cMed_d = widths.cMed_d
        self.cMed_u = widths.cMed_u
        self.cMed_s = widths.cMed_s
        self.cMed_c = widths.cMed_c
        self.cMed_b = widths.cMed_b
        self.cMed_t = widths.cMed_t
        self.clep = widths.clep
        self.whad = widths.whad
        self.wqcd = widths.wqcd
        self.widthshad = widths.widthshad
        self.widthmu = widths.widthslep["muon"]
        self.widthspert = widths.widthspert
        self.masses = widths.masses
        self.mhad = widths.mhad
        self.Rtotal=[]
        self.Rhad = {}
        self.Rpert = []
        
    # non-perturbative
    def hadronic(self):
        self.Rtotal=[]
        self.Rhad = {}
        # initialize R dict
        for i in range(0, len(ch.labelhad)):
            self.Rhad[ch.label[i]] = []
        # calc R values
        for i in range(0, len(self.masses)):
            self.Rtotal.append(self.intdiv(self.whad[i],self.widthmu[i]))
            for channel in ch.labelhad:
                num = self.widthshad[channel][i]
                den = self.widthmu[i]
                self.Rhad[channel].append(self.intdiv(num,den))

    def perturbative(self):
        self.Rpert = []
        for i in range(0,len(self.masses)):
            gammaMuSM = self.GammaVff(1, 1, self.masses[i], -1, par.mlep_[1])
            rpval = self.RInclusive(self.masses[i])*self.intdiv(gammaMuSM,self.widthmu[i])
            self.Rpert.append(rpval)
    
    # function to save Rvalues
    def calc(self):
        if self.clep[1]==0:
            print ("The R-value calculation works only for non-zero values for the model coupling with muons.")
            return
        self.hadronic()
        self.perturbative()
        self.Rmax = max(self.Rtotal)
        self.Rmin = min([i for i in self.Rtotal[int(len(self.Rtotal)/7):] if i != 0])
        
    # function to plot Rvalues
    #---------------  plot -------------------# 
    def plot(self,xrange=[0.3,2.], yrange =None, name= None,Rsingle=[]):
        Wtype= None
        if self.DM != "No":
            Wtype = 'DM_'+self.DM.replace(" ", "_")
        elif self.DM=="No":
            Wtype = 'SM'
            
        if yrange==None:
            yrange = [self.Rmin*5,self.Rmax*5]
   
        fig, ax = fig, ax = plt.subplots(figsize=(9., 4.8))
        if max(data.RPDG[:880])>self.Rmin*5:
            ax.errorbar(data.massPDG,data.RPDG, [data.low_eRPDG,data.up_eRPDG] , color ='black', ls ='none', fmt = '.', ms =2.5
                        ,fillstyle='full',elinewidth=0.5, capsize=0.5,capthick = 0.3, label= "PDG data 2020", zorder = -5)
        
        ax.plot(self.masses,self.Rtotal, c='darkorange', lw =1.3, label='all hadrons')
        if len(Rsingle)>0:
            for xR in Rsingle:
                ax.plot(self.masses,self.Rhad[xR], lw=1.2, label=xR)
        ax.plot(self.masses,self.Rpert, c='forestgreen', lw =1.2, label='quarks+QCD corrections')
        ax.set_title(self.modelname)
        ax.set_xlabel("$\\sqrt{s}$ [GeV]",fontfamily= 'serif',fontsize=14)
        ax.set_ylabel("$R^{\\mathcal{H}}_{\mu} = \\frac{\\Gamma\;(Z_Q \\; \\to \\; \\mathrm{hadrons})}{\\Gamma\;({Z_Q \\to \\; \\mu^+\\mu^-)}}$",fontfamily= 'serif',fontsize=14)
        ax.legend(loc="best",ncol=2)
        
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        plt.yscale("log")
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.2)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if name:
            try:
                plt.savefig('models/'+self.modelname.replace(" ","_")+"_"+Wtype+'/plots/'+name)
            except:
                print ("Please choose a valid name+extension, e.g. plot(name='plot_1.pdf').")
        plt.show()
        plt.close()
        return fig
        
            
    def save(self):
        Rtype = None
        if self.DM !="No":
            Rtype = 'DM_'+self.dmtype.replace(" ", "_")
        if self.DM =="No":
            Rtype = 'SM'
        if not Rtype==None:
            mstr = self.modelname.replace(" ", "_")
            for ichannel in ch.labelhad:
                path = 'models/'+mstr+'_'+Rtype+'/Rvalues/R_'+ichannel+'.txt'
                self.writefiles(path,self.masses,self.Rhad[ichannel])
            self.writefiles('models/'+mstr+"_"+Rtype+'/Rvalues/'+mstr+'_Rratio_total.txt',x=self.masses,y=self.Rtotal)
        else:
            print("DM type not specified")
    