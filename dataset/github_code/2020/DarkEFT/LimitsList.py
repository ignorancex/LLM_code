#!/usr/bin/env python
# -*- coding: utf-8 -*-



################################################## ##############################
#########################  List of implemented limits  #########################
######################### ######################### #############################

### Define a class to store all the relevant data about the external limits 
### as well as the way of recasting them to the fermion portal case. Then loads all the implemented exeternal limits


### L. Darme, S. Ellis, T. You, 29/12/2019




############################ Messy library import part #########################
import numpy as np

# Importing the other sub-Modules

import UsefulFunctions as uf
import Production as br
import Detection as de

verbose=True


class Limit:
    """
    Contain the informations to recast a given experimental limit
    
    exp:     The experiment associated to the limit
    
    channel: The type of limit (e.g MissingE, Decay ...)
    
    descr:   Any relevant additional details on the limit 
    
    ref:     The Inspire reference for the limit
    
    mx_ini:  The range of Mx value on which the limit is relevant (by convention 
             this is typically the mass of the lower state X1 in case there is
             two dark sector states, such as for decay limits)
             
    lim_ini: The actual limit (typically from a dark photon model)
    
    lim_inilist: Dictionnary with the initial limits stored in (mx, lim) list for different operator in case they exist ( "V" and "AV")
    """

    def __init__(self, name, exp, channel, Delini,interptype="log"):
        self.exp = exp
        self.channel = channel
        self.delini = Delini
        self.descr = "No description for this limit"
        self.ref = "No reference for this limit"
        self.combthr=1.1 # Standard value used when relevant to combine the upper and lower limit      
        try: 
            self.name = name
            self.mx_ini, self.lim_ini = uf.LoadData('LimData/'+name+'.dat',interptype)   
#             if verbose: print("Loading: ",  self.name,self.lim_ini   )             
        except: 
            self.name = "NotDefined"
            self.mx_ini = np.logspace(-3., 1., 30.)
            self.lim_ini = np.zeros(np.shape(self.mx_ini ))    

        try: 
            self.model = model
        except: 
            self.model = "No model for this limit"
  
        self.lim_inifull = {"V":(self.mx_ini,self.lim_ini),"AV":(self.mx_ini,self.lim_ini)}
    
    def UpdateLimIni(self,mx,lim,optype="V"): 
#         print(optype,(mx,lim)) 
        self.lim_inifull.update({optype:(mx,lim)})
        if optype=="V":
            self.lim_ini = lim
            self.mx_ini = mx
            
        
    def recast(self,delrec, geff, optype,useHeavyMeson=False):
        # we define the recast based on the expected signal
        
        if self.channel == "decay":
            if self.model == "EFT": # For naive NoE based limits
                xi,EffLimtmp = de.GetNaiveDecayLimits( delrec,  self.exp ,10,geff,optype)
                xi,EffLim,EffLim_low = de.RecastDecayLimit(xi, EffLimtmp, delrec,delrec,self.exp,geff,optype)
                xi_full, Lim_full=uf.CombineUpDown(xi,EffLim_low,EffLim,self.combthr)        
            else: # Standard recasting case
                xi_full, Lim_full=de.FastDecayLimit(self.exp,self.mx_ini, self.lim_ini , self.delini, delrec, geff, optype,self.combthr,useHeavyMeson)
        elif self.channel == "heavymesondecay": 
            if self.model == "EFT":
                xi,EffLimtmp = de.GetNaiveDecayLimits( delrec, self.exp ,10,geff,optype,True)
                xi_heavy,EffLim_heavy,EffLim_heavy_low = de.RecastDecayLimit(xi,EffLimtmp , delrec,delrec,self.exp,geff,optype,True)
                xi_full, Lim_full=uf.CombineUpDown(xi_heavy,EffLim_heavy_low,EffLim_heavy,self.combthr)
            else:
                xi_full, Lim_full=de.FastDecayLimit(self.exp,self.mx_ini, self.lim_ini , self.delini, delrec, geff, optype,self.combthr,useHeavyMeson)
        elif self.channel=="scattering":
            xi_full, Lim_full=de.FastScatLimit(self.exp,self.mx_ini, self.lim_ini , self.delini,delrec, geff, optype)
        elif self.channel=="monogam":
            xi_full, Lim_full=de.FastMonoPhoton(self.exp,self.mx_ini, self.lim_ini , self.delini,delrec, geff, optype)
        elif self.channel=="invisibledecayBmtoKm":
            if self.exp == "babar":
                xi_full, Lim_full=  de.FastInvMesDecay("babar_BmtoKmnunu",delrec, geff, optype)
            elif self.exp == "belle2":
                xi_full, Lim_full=  de.FastInvMesDecay("belle2_BmtoKmnunu",delrec, geff, optype)
        elif self.channel=="invisibledecayBmtoPim":
            xi_full, Lim_full=  de.FastInvMesDecay("babar_BmtoPimnunu",delrec, geff, optype)
        elif self.channel == "invisibledecayB0toPi0":
            xi_full, Lim_full=  de.FastInvMesDecay("belle_B0toPi0nunu",delrec, geff, optype)
        elif self.channel == "invisibledecayB0toK0":
            xi_full, Lim_full=  de.FastInvMesDecay("belle_B0toK0nunu",delrec, geff, optype)
        elif self.channel=="monojet_down":
            xi_full, Lim_full = de.FastMonoJet(self.exp,self.mx_ini, self.lim_ini , self.delini,delrec, geff, optype)
        elif self.channel=="monojet_up":
            xi_full, Lim_full = de.FastMonoJet(self.exp,self.mx_ini, self.lim_ini , self.delini,delrec, geff, optype)
        elif self.channel == "invisibledecayKL0toPi0":
            if self.exp == "e391a":
                xi_full, Lim_full=  de.FastInvMesDecay("e391a_KL0toPi0nunu",delrec, geff, optype)
            if self.exp == "na62":
                xi_full, Lim_full=  de.FastInvMesDecay("na62_KL0toPi0nunu",delrec, geff, optype)
        elif self.channel == "invisibledecayPi0":
            xi_full, Lim_full=  de.FastInvMesDecay("na62_pi0toinvisible",delrec, geff, optype)
        elif self.channel == "invisibledecayJPsi":
            xi_full, Lim_full=  de.FastInvMesDecay("bes_JPsitoinvisible",delrec, geff, optype)
        elif self.channel == "invisibledecayUpsilon":
            xi_full, Lim_full=  de.FastInvMesDecay("babar_Upsilontoinvisible",delrec, geff, optype)   
        elif self.channel == "invisibledecayKptoPip":
            if self.exp == "na62":
                xi_full, Lim_full=  de.FastInvMesDecay("na62_KptoPipa",delrec, geff, optype)  
            elif self.exp == "e949":
                xi_full, Lim_full=  de.FastInvMesDecay("e949_KptoPipa",delrec, geff, optype)  
        elif self.channel == "cosmicrays":
            xi_full, Lim_low_full, Lim_high_full = de.FastCRLimit("t2k", delrec, geff,optype)
            xi_full, Lim_full = uf.CombineUpDown(xi_full, Lim_low_full, Lim_high_full)
        elif self.channel == "low_cooling":  
            xi_full, Lim_full= de.FastSN1987Limit(self.lim_inifull ,delrec, geff, optype,False)
        elif self.channel == "high_cooling": 
            xi_full, Lim_full= de.FastSN1987Limit( self.lim_inifull ,delrec, geff, optype,True)
        else:
            print("Channel selected: ", self.channel, " is not currently implemented. Possible choices: \n",
            "'decay' : faser, mathusla, ship, charm, seaquest, seaquest_Phase2, lsnd \n",
            "heavymesondecay : ship"
            "'scattering' :  nova, miniboone, sbnd \n",
            "'missingE' : babar, belleII, atlas, lep \n",
             "'cosmicrays' : t2k (decay from cosmic ray showers into t2k), ",
             "cooling : for sn1987_low (pessimistic limit from SN1987 cooling), sn1987_high (optimistic limit from SN1987 cooling) ",
             "Invisible light meson decays: na62 (pi0decay and invisibledecayKptoPip) e949 (pi0decay and invisibledecayKptoPip), e391a (invisibledecayKL0toPi0)",
             "Invisible heavy meson decay: belle (invisibledecayB0toPi0 ,invisibledecayB0toK0) and belleII (invisibledecayBmtoKm)")
            xi_full= np.logspace(-3., 1., 30.);
            Lim_full =np.zeros(np.shape(xi_full))   
        return xi_full,Lim_full  

AllLimits={} # Dictionnary for all the limits
 
def UpdateLimitList(limit,name="noname"):
    
    if name=="noname":
        name=limit.exp+ "_"+limit.channel # We fill up the name automatically if not given

    if name in AllLimits: # Making a new limits if one already existing for the same 
#         print("Test:", name, AllLimits)
        i = 1
        name_new=name+"_"+str(i)
        while (name_new in AllLimits):
            i += 1
            name_new=name+"_"+str(i)
        name=name_new
    AllLimits.update({name:limit})


## This function runs over a list of experiments and search channels,
## compute the recasted bound, print it to file and return everything in a dictionnary with a list of label
## Note that it can also just read an external file and use it directly as the limit

def GetLimits(LimList,Del,geff,optype="V",PrintToFile = False, ReadFromFile=False, filename=''):
    
    Res = {}
    LabelLimit = []

    # Making sure the  input for the effective couplings is okay
    if type(geff) is dict:
        ge=geff
    else:
        if np.ndim(geff[0]) >0:
            geffdiag=(geff[0][0][0],geff[1][0][0],geff[2][0][0])  
            gMesDecay=(geff[1][2][1],geff[1][2][1],geff[1][2][0],geff[1][2][0],geff[1][1][0],geff[1][1][0]) # gd32, gd32,gd31,gd31,gd21,gd21
        else:
            geffdiag=  geff
            gMesDecay=(0,0,0,0,0,0)
        ge={"gu11":geffdiag[0],"gd11":geffdiag[1],"gl11":geffdiag[2],"gl22":geffdiag[2] \
        ,"gd32":gMesDecay[0],"gd31":gMesDecay[2],"gd21":gMesDecay[4]}#We make the couping a dictionnary to have more freedom

    resultsDict = {} #e.g. {"exp1_channel1":(mxlist, limlist), "exp2_channel2":{mxlist, limlist}, ... }
    expchannelList = [] #e.g. ["exp1_channel1", "exp2_channel2", etc.]

    for expchannel in LimList: 
        if not expchannel in AllLimits:
            print( "Selected limit " + expchannel +" not available, choose from: ", AllLimits.keys())
#             raise Exception()
        else:
            if ReadFromFile: 
                if filename == '': 
                    filestr = 'Output/Lim_'+expchannel+".dat"
                else:
                    filestr = filename
                filedat = np.loadtxt(filestr)
                Mx, LamLim = np.transpose(filedat)
            else: 
                #print(exp, channel)
    
                Mx, LamLim = AllLimits[expchannel].recast(Del,ge,optype)
                
    
            expchannelList.append(expchannel)
            resultsDict[expchannel] = (Mx,LamLim)
    
            if PrintToFile:
                if filename == '':
                    filestr = 'Output/Lim_'+expchannel+".dat"
                else: 
                    filestr = filename
                np.savetxt(filestr, np.transpose(resultsDict[expchannel]))

    return resultsDict, expchannelList  
        


################################################## #########################
#########################                           #########################
#########################  Loading external limits  #########################
#########################                           #########################
######################### ######################### #########################

geffem={"gu11":2/3.,"gd11":-1/3.,"gl11":-1.}

print("Loading limits:")

#########################  Scattering #########################
if verbose: print("Scattering ...")

miniboone_scattering=Limit("miniboone_1807.06137","miniboone","scattering",0.)
miniboone_scattering.descr="""
Limits from MiniBooNE collaboration for light dark matter scattering, produced at the beam dump. 

Extracted from Figure 24.a, the data is given as epsilon^2, for alpha_D = 0.5, 
we rescale it by 5 ^1/4. since the scattering limit scale as eps^4 alpha_D
"""
miniboone_scattering.ref="inspirehep.net/record/1682906"
miniboone_scattering.UpdateLimIni(miniboone_scattering.mx_ini,np.sqrt(miniboone_scattering.lim_ini)*np.power(5,1/4.) )
UpdateLimitList(miniboone_scattering)

############

sbnd_scattering=Limit("sbnd_1609.01770","sbnd","scattering",0.)
sbnd_scattering.descr="""
Projective Limits from SBND collaboration for light dark matter scattering
produced by 1609.01770, for 2. * 10^20  PoT, based on 10 events reach

Extracted from Figure 9.b, the data is given as epsilon^2, for alpha_D = 0.5
"""
sbnd_scattering.ref="inspirehep.net/record/1485563"
sbnd_scattering.UpdateLimIni(sbnd_scattering.mx_ini,np.sqrt(sbnd_scattering.lim_ini)*np.power(5,1/4.) )
UpdateLimitList(sbnd_scattering)

############

ship_scattering=Limit("ship_1609.01770","ship","scattering",0.)
ship_scattering.descr="""
Projective limits from SHIP collaboration for light dark matter scattering,
produced by  1609.01770, for 2. * 10^20  PoT, based on 10 events reach

Extracted from Figure 24.a, the data is given as epsilon^2, for alpha_D = 0.5
"""
ship_scattering.ref="inspirehep.net/record/1485563"
ship_scattering.UpdateLimIni(ship_scattering.mx_ini,np.sqrt(ship_scattering.lim_ini)*np.power(5,1/4.) )
UpdateLimitList(ship_scattering)


############

nova_scattering=Limit("nova_1807.06501","nova","scattering",0.)
nova_scattering.descr="""
Limit produced by 1807.06501 based on NOvA collaboration neutrino-electron scattering 
in 1710.03428, recasted for light dark matter scattering for 2.97 * 10^20  PoT.
This is not a projection, limit for 58 events, although the authors of 1807.06501 merely
call for a full analysis by the collaboration

Extracted from Figure 2, with alpha_D = 0.05, the data is given as y = eps^2 * alpha_D *(mx/ mv)^4= 0.00062 eps^2
"""
nova_scattering.ref="inspirehep.net/record/1682772"
# print("nova",nova_scattering.mx_ini,nova_scattering.lim_ini)
nova_scattering.UpdateLimIni(nova_scattering.mx_ini,np.sqrt(nova_scattering.lim_ini)/np.sqrt(0.00062)/np.power(2,1/4.) )
UpdateLimitList(nova_scattering)
# print("nova2",nova_scattering.mx_ini,nova_scattering.lim_ini)

#########################  Long-lived particles #########################
if verbose: print("Long-lived dark sector ...")

############ Based on: Kling, 2018 MATHUSLA

mathusla_decay=Limit("mathusla_1810.01879","mathusla","decay",0.1)
mathusla_decay.descr="""
Projective limits for MATHUSLA from 1810.01879, with 3 ab-1 of luminosity from HL-LHC

Energy cut > 0.6 GeV
"""
mathusla_decay.ref="inspirehep.net/record/1696950"
mathusla_decay.combthr=1.2
UpdateLimitList(mathusla_decay)

############ Based on: Kling, 2018 FASER

faser_decay=Limit("faser_1810.01879","faser","decay",0.1)
faser_decay.descr="""
Projective limits for FASER from 1810.01879,  with 3 ab-1 of luminosity from HL-LHC

Energy cut > 100 GeV
"""
faser_decay.ref="inspirehep.net/record/1696950"
faser_decay.combthr=1.1
UpdateLimitList(faser_decay)


############ Based on: 2018 SeaQuest phase 1 and 2

seaquest_phase2_decay=Limit("seaquest_phase2_1804.00661","seaquest_phase2","decay",0.1)
seaquest_phase2_decay.descr="""
Projective limits 1804.00661 Figure 12, 10 events limits with 5m−6m fiducial decay region Phase 2 with 10^20 PoT

"""
seaquest_phase2_decay.ref="inspirehep.net/record/1665691"
UpdateLimitList(seaquest_phase2_decay)


seaquest_phase1_decay=Limit("","seaquest_phase1","decay",0.1)
seaquest_phase1_decay.descr="""
Projective limit, 1804.00661, rescaled from Figure 12 given the reduced number of PoT: 1.44×10^18 PoT 
, 10 events limits with 5m−6m fiducial decay regions 
"""
seaquest_phase1_decay.ref="inspirehep.net/record/1665691"
seaquest_phase1_decay.UpdateLimIni(seaquest_phase1_decay.mx_ini,np.sqrt(seaquest_phase1_decay.lim_ini)/np.power(0.014,1/4.))
UpdateLimitList(seaquest_phase1_decay)

############ Based on: Izaguiire, 2017 LSND (on-shell pi0 only?)

lsnd_decay_2=Limit("lsnd_1703.06881","lsnd","decay",0.1)
lsnd_decay_2.descr="""
Limits for LSND collaboration, based on  original neutrino scattering search from hep-ex/0104049, 
recasted for iDM in 1703.06881 Fig 6, with splitting 10%
"""
lsnd_decay_2.ref="inspirehep.net/record/1682906"
UpdateLimitList(lsnd_decay_2,"lsnd_decay_2")


############  LSND ---- based Darme 2018
 
lsnd_decay=Limit("lsnd_1807.10314","lsnd","decay",0.15)
lsnd_decay.descr="""
Limits for LSND collaboration, based on  original neutrino scattering search from hep-ex/0104049, 
recasted for iDM in 1807.10314 Fig5a, with splitting 15%
"""
lsnd_decay.ref="inspirehep.net/record/1684267"
UpdateLimitList(lsnd_decay)

############  CHARM ---- based Tsai 2019
 
charm_decay=Limit("charm_1908.07525","charm","decay",0.1)
charm_decay.descr="""
Limits for CHARM collaboration, based on  original neutrino scattering search from Phys.Lett. 128B (1983) 361,
 recasted for iDM in 1908.07525 Fig1.c

Ecut > 3 GeV
"""
charm_decay.ref="inspirehep.net/record/1682906"
UpdateLimitList(charm_decay)

############  SHIP decay prediction
 
ship_decay=Limit("","ship","decay",0.1)
ship_decay.descr="""
Naive limit prediction for SHIP, based on the routine for production and decay implemented in this code.
Does not include the experimental efficiency. The output is the 10 events line.
"""
ship_decay.ref=""
ship_decay.model="EFT"
UpdateLimitList(ship_decay)

ship_heavymesondecay=Limit("","ship","heavymesondecay",0.1)
ship_heavymesondecay.descr="""
Naive limit prediction for SHIP, based on the routine for production and decay implemented in this code.
Does not include the experimental efficiency. The output is the 10 events line.
Production is generated by heavy meson decay
"""
ship_heavymesondecay.ref=""
ship_heavymesondecay.model="EFT" # The generated limits are already in the EFT
UpdateLimitList(ship_heavymesondecay)

#########################  Missing energy searches  #########################
if verbose: print("Mono-X searches ...")

############  BaBAr  # Based on Essig 2013

babar_monogam=Limit("babar_1309.5084","babar","monogam",0.0)
babar_monogam.descr="""
Limits for BaBar collaboration, based on the recasting from 1309.5084 of the upsilon decay 
mono-photon search from 0808.0017.

Notice that the absence of "bump" in the reconstructed dark sector invariant mass distribution
significantly weakens the reach. Better control of the background could lead to improvement
"""
babar_monogam.ref="inspirehep.net/record/1254859,inspirehep.net/record/792059"
UpdateLimitList(babar_monogam)

############  Belle II  # Based on Essig 2013

belle2_monogam=Limit("belle2_1309.5084","belle2","monogam",0.0)
belle2_monogam.descr="""
Limits for Belle II collaboration, based on the proejction from 1309.5084

Notice that the absence of "bump" in the reconstructed dark sector invariant mass distribution
significantly weakens the reach. Better control of the background could lead to improvement
"""
belle2_monogam.ref="inspirehep.net/record/1254859"
UpdateLimitList(belle2_monogam)


#########################  MonoJet searches at ATLAS 35.9 fb-1  #########################
data =  np.loadtxt('Data/LimData/atlas_1711.03301.dat')
xi_ATLAS = np.abs(data[:,0])
Lim_ATLAS_Down = data[:,1]
Lim_ATLAS_Up= data[:,2]

atlas_monoXlow=Limit("","atlas","monojet_down",0.0)
atlas_monoXlow.descr="""
Limits for ATLAS collaboration, 1711.03301 based on the upper recasted limit from 1807.03817 
and our own recast.
"""
atlas_monoXlow.ref="inspirehep.net/record/1635274"
atlas_monoXlow.UpdateLimIni(xi_ATLAS,Lim_ATLAS_Down)
UpdateLimitList(atlas_monoXlow)


atlas_monoXhigh=Limit("","atlas","monojet_up",0.0)
atlas_monoXhigh.descr="""
Limits for ATLAS collaboration, 1711.03301 based on the upper recasted limit from 1807.03817 
and our own recast.
"""
atlas_monoXhigh.ref="inspirehep.net/record/1635274"
atlas_monoXhigh.UpdateLimIni(xi_ATLAS,Lim_ATLAS_Up)
UpdateLimitList(atlas_monoXhigh)


#########################  MonoPhoton searches at DELPHI from 1103.0240  #########################

xi_LimLEP_V,LimLEP_V  = uf.LoadData('LimData/lep_1103.0240_V.dat',"lin")
xi_LimLEP_AV,LimLEP_AV  = uf.LoadData('LimData/lep_1103.0240_AV.dat',"lin")

lep_monoX=Limit("lep_1103.0240","lep","monogam",0.0)
lep_monoX.descr="""
Limits for LEP collaboration, based on the recasting from 1103.0240 for the upper limit. 
The lower limit is given by the breakdown of the EFT at the LEP CoM energy.
"""
lep_monoX.ref="inspirehep.net/record/890992"
lep_monoX.UpdateLimIni(xi_LimLEP_V,LimLEP_V)
lep_monoX.UpdateLimIni(xi_LimLEP_AV,LimLEP_AV,"AV")
UpdateLimitList(lep_monoX)

######################### SN1987 cooling rate constraints #####
if verbose: print("SN1987 cooling ...")

sn1987low_cooling=Limit("sn1987low","sn1987","low_cooling",0.0,"lin")
sn1987low_cooling.descr="""
Lower limit from SN1987 cooling constraints, recasted from ***
"""
sn1987low_cooling.ref="inspirehep.net/record/1682906"
newmx=np.power(10.,sn1987low_cooling.mx_ini/1.)
newlim=np.power(10.,sn1987low_cooling.lim_ini/1.)
sn1987low_cooling.UpdateLimIni(newmx,newlim,"V")
sn1987low_cooling.UpdateLimIni(newmx,newlim,"AV")
UpdateLimitList(sn1987low_cooling)
# print(sn1987low_cooling.mx_ini,sn1987low_cooling.lim_ini)


sn1987high_cooling=Limit("sn1987high","sn1987","high_cooling",0.0,"lin")
sn1987high_cooling.descr="""
Upper limit from SN1987 cooling constraints, recasted from ***

In the case of AV coupling, we use instead the lower limit on the pi0->invisible BR
derived in *** from sn1987 cooling
"""
sn1987high_cooling.ref="inspirehep.net/record/1682906"
xi_LimSN_AV_up,LamlimSN_AV_up= uf.LoadData('LimData/sn1987_pi0decay.dat',"log")
sn1987high_cooling.UpdateLimIni(np.power(10.,sn1987high_cooling.mx_ini/1.),np.power(10.,sn1987high_cooling.lim_ini/1.))
sn1987high_cooling.UpdateLimIni(xi_LimSN_AV_up,LamlimSN_AV_up,"AV")
UpdateLimitList(sn1987high_cooling)

# print(sn1987high_cooling.mx_ini,sn1987high_cooling.lim_ini)

#########################  Invisible decay of pi0  #################

if verbose: print("Invisible meson decay ...")

na62_invisibledecayPi0=Limit("na62_talkKaon2019","na62","invisibledecayPi0",0.0)
na62_invisibledecayPi0.descr="""
Limits for invisible decay branching ratio of pi0 meson as constrained by the NA62 collaboration

Currently value only from https://indico.cern.ch/event/769729/sessions/318725/ Ruggiero's talk
BR < 4.4 10^-9 at 90% CL
"""
na62_invisibledecayPi0.ref="indico.cern.ch/event/769729/contributions/3510938/attachments/1905346/3146619/kaon2019_ruggiero_final.pdf"
UpdateLimitList(na62_invisibledecayPi0)

#########################  Invisible decay of heavy mesons  #################
if verbose: print("Invisible heavy meson decay ...")

babar_invisibledecayBmtoKm=Limit("babar_invisibledecayBmtoKm","babar","invisibledecayBmtoKm",0.0)
belle2_invisibledecayBmtoKm=Limit("belle2_invisibledecayBmtoKm","belle2","invisibledecayBmtoKm",0.0)
belle_invisibledecayB0toPi0=Limit("belle_invisibledecayB0toPi0","belle","invisibledecayB0toPi0",0.0)
belle_invisibledecayB0toK0=Limit("belle_invisibledecayB0toK0","belle","invisibledecayB0toK0",0.0)
babar_invisibledecayBmtoPim=Limit("babar_invisibledecayBmtoPim","babar","invisibledecayBmtoPim",0.0)
e391a_invisibledecayKL0toPi0=Limit("e391a_invisibledecayKL0toPi0","e391a","invisibledecayKL0toPi0",0.0)
na62_invisibledecayKL0toPi0=Limit("na62_invisibledecayKL0toPi0","na62","invisibledecayKL0toPi0",0.0)
e949_invisibledecayKptoPip=Limit("e949_invisibledecayKptoPip","e949","invisibledecayKptoPip",0.0)
na62_invisibledecayKptoPip=Limit("na62_invisibledecayKptoPip","na62","invisibledecayKptoPip",0.0)


bes_invisibledecayJPsi=Limit("bes_invisibledecayJPsi","bes","invisibledecayJPsi",0.0)
bes_invisibledecayJPsi.descr="Limit BR JPSi-> inv < 7.2 * 10^-4 from BES collaboration  0710.0039"

babar_invisibledecayUpsilon=Limit("babar_invisibledecayUpsilon","babar","invisibledecayUpsilon",0.0)
babar_invisibledecayUpsilon.descr="Limit BR Upsilon-> inv < 3.0 * 10^-4 from BABAR collaboration  0908.2840"


UpdateLimitList(babar_invisibledecayBmtoKm);UpdateLimitList(belle2_invisibledecayBmtoKm);UpdateLimitList(belle_invisibledecayB0toPi0);
UpdateLimitList(belle_invisibledecayB0toK0);UpdateLimitList(babar_invisibledecayBmtoPim);
UpdateLimitList(e391a_invisibledecayKL0toPi0);UpdateLimitList(na62_invisibledecayKL0toPi0);
UpdateLimitList(e949_invisibledecayKptoPip);UpdateLimitList(na62_invisibledecayKptoPip)
UpdateLimitList(bes_invisibledecayJPsi);UpdateLimitList(babar_invisibledecayUpsilon)



######################### CR limits at T2K ################# Currently not used
# if verbose: print("Decay from CR production limits ...")
# 
# t2k_cosmicrays=Limit("t2k_cosmicrays","t2k","cosmicrays",0.0)
# t2k_cosmicrays.descr="""
# Projections from T2K limits based on cosmic ray production of dark sector states
# limits for 500 number of signal events per year, neutrino background for superK is ~150
# 
# """
# UpdateLimitList(t2k_cosmicrays)

