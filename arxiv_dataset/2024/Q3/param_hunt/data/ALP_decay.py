# this performs the same task as ALP_decay.py but with varying number of events
# for efficiency it miht make sense to generate several events per parameter point and cut afterward rather then generate 3, 5, 7...
# format of a single sample with n_obs [m_alp, tau_alp, features_0, features_1, ... features_(nobs-1)] (as a ditionary with different keys for different i_obs)


import numpy as np
import random

# this means that the geometry is a pyramid frustum with ship size
ship = 1

Pbeam = 400 # in GeV
mp = 0.93827 # in GeV
Ebeam = np.sqrt(mp**2 + Pbeam**2) # in GeV

mB0 = 5.27963 # in GeV
mB = 5.27932 # in GeV
mK0 = 0.49761 # in GeV
mK = 0.49367 # in GeV
mK0star = 0.89555 # in GeV
mKstar = 0.89176 # in GeV



def lambda_abc(aa, bb, cc): # Phase-space factor for decay
  return (aa**2 - (bb + cc)**2) * (aa**2 - (bb - cc)**2)

raw_data = np.loadtxt("beauty_100kEvts_pp_8.2_400GeV_ptHat300MeV_new_ok.txt")[:,2:6] # Pythia events
b_mesons = raw_data[(np.abs(raw_data[:,0])==511) | (np.abs(raw_data[:,0])==521)]     # Select B mesons 

normpB = len(b_mesons)/1.e5 # Normalisation factor (not currently needed)

def b_meson_decay(b_meson, m_alp): # Simulates a random decay of a given B meson into K + a. At the moment decays into K* + a are not implemented.

  id = b_meson[0]
  p_b = np.array([b_meson[1],b_meson[2],b_meson[3]])

  if(np.abs(id) == 511):
    massB = mB0
    massK = mK0
  else:
    massB = mB
    massK = mK

  pp2 = np.sum(p_b**2)

  energy = np.sqrt(massB**2 + pp2)

  # Determine Lorentz transformation from cms frame to lab frame
  
  gamma = energy/massB
  beta = p_b/energy

  Lambda = np.array([[ gamma, gamma*beta[0], gamma*beta[1], gamma*beta[2]], 
            [ gamma*beta[0], 1 + (gamma - 1)*beta[0]**2/(pp2/energy**2), (gamma - 1)*(beta[1]*beta[0])/(pp2/energy**2), (gamma - 1)*(beta[2]*beta[0])/(pp2/energy**2)],
            [ gamma*beta[1], (gamma - 1)*beta[0]*beta[1]/(pp2/energy**2), 1 + (gamma - 1)*(beta[1]**2)/(pp2/energy**2), (gamma - 1)*(beta[2]*beta[1])/(pp2/energy**2)],
            [ gamma*beta[2], (gamma - 1)*beta[0]*beta[2]/(pp2/energy**2), (gamma - 1)*(beta[1]*beta[2])/(pp2/energy**2), 1 + (gamma - 1)*(beta[2]**2)/(pp2/energy**2)]])

  # Generate decay in cms frame

  pa2cm = lambda_abc(massB, massK, m_alp)/(4*massB**2)

  thetaacm = np.arccos(random.uniform(-1,1))
  phiacm = random.uniform(0, 2*np.pi)

  p4acm = np.array([ np.sqrt(pa2cm + m_alp**2), np.sqrt(pa2cm)*np.sin(thetaacm)*np.cos(phiacm), np.sqrt(pa2cm)*np.sin(thetaacm)*np.sin(phiacm), np.sqrt(pa2cm)*np.cos(thetaacm)])
  
  # Boost to lab frame
  
  p4a = np.dot(Lambda, p4acm)

  return p4a

def alp_decay(p4a, m_alp, ctau_alp, z_min, z_max, z_cal, l_x, l_y, dr_gg, E_min): # Simulates a random decay of a given ALP with mass m_alp and decay length ctau_alp into two photons
                                                                                    # It is possible to force the decay to happen between z_min and z_max (at the cost of reducing the event weight)
                                                                                    # l_x, l_y, displ_y are the detector x, y size and the possible y displacement (relative to the center of the detector)
                                                                                    # dr_gg is the minimum photon separation and E_min the minimum photon energy (event selection criteria) 
  E_alp = p4a[0]
  p3_alp = p4a[1::]
  p_alp = np.sqrt(np.sum(p3_alp**2))

  # decay position is related to z_max, not z_cal
  l_min = p_alp / p3_alp[2] * z_min
  l_max = p_alp / p3_alp[2] * z_max

  decayLength_alp = ctau_alp * p_alp / m_alp
  weight = np.exp(-l_min/decayLength_alp)-np.exp(-l_max/decayLength_alp)

  decay_distance =   l_min - decayLength_alp * np.log(1 + (np.exp((l_min - l_max)/decayLength_alp) - 1) * random.uniform(0,1)) # Generates random draw from exponential distribution
  decay_position = decay_distance / p_alp * p3_alp

  # Determine Lorentz transformation from cms frame to lab frame

  gamma = E_alp/m_alp
  beta = p3_alp/E_alp

  Lambda = np.array([[ gamma, gamma*beta[0], gamma*beta[1], gamma*beta[2]], 
            [ gamma*beta[0], 1 + (gamma - 1)*beta[0]**2/(p_alp**2/E_alp**2), (gamma - 1)*(beta[1]*beta[0])/(p_alp**2/E_alp**2), (gamma - 1)*(beta[2]*beta[0])/(p_alp**2/E_alp**2)],
            [ gamma*beta[1], (gamma - 1)*beta[0]*beta[1]/(p_alp**2/E_alp**2), 1 + (gamma - 1)*(beta[1]**2)/(p_alp**2/E_alp**2), (gamma - 1)*(beta[2]*beta[1])/(p_alp**2/E_alp**2)],
            [ gamma*beta[2], (gamma - 1)*beta[0]*beta[2]/(p_alp**2/E_alp**2), (gamma - 1)*(beta[1]*beta[2])/(p_alp**2/E_alp**2), 1 + (gamma - 1)*(beta[2]**2)/(p_alp**2/E_alp**2)]])

  # Generate decay in cms frame

  E_gamma_cm = m_alp / 2

  thetaacm = np.arccos(random.uniform(-1,1))
  phiacm = random.uniform(0, 2*np.pi)

  p4_gamma1_cm = np.array([ E_gamma_cm, E_gamma_cm*np.sin(thetaacm)*np.cos(phiacm), E_gamma_cm*np.sin(thetaacm)*np.sin(phiacm), E_gamma_cm*np.cos(thetaacm)])
  p4_gamma2_cm = np.array([ E_gamma_cm, - E_gamma_cm*np.sin(thetaacm)*np.cos(phiacm), - E_gamma_cm*np.sin(thetaacm)*np.sin(phiacm), - E_gamma_cm*np.cos(thetaacm)])

  # Boost to lab frame

  p4_gamma1 = np.dot(Lambda, p4_gamma1_cm)
  p4_gamma2 = np.dot(Lambda, p4_gamma2_cm)

  # Order photons such that E_gamma1 >= E_gamma2

  if(p4_gamma1[0] < p4_gamma2[0]):
    p4_gammatemp = p4_gamma1
    p4_gamma1 = p4_gamma2
    p4_gamma2 = p4_gammatemp

  # Propagate photons to z_cal

  travel_distance = z_cal - decay_position[2]

  hit_gamma1 = travel_distance / p4_gamma1[3] * p4_gamma1[[1,2,3]] + decay_position
  hit_gamma2 = travel_distance / p4_gamma2[3] * p4_gamma2[[1,2,3]] + decay_position

  # Fill event dictionary

  event = {
    "gamma_1_calo_hit": hit_gamma1,
    "gamma_1_4momentum": p4_gamma1,
    "gamma_2_calo_hit": hit_gamma2,
    "gamma_2_4momentum": p4_gamma2,
    "decay_position": decay_position,
    "event_weight": weight
  }
  
  
  # Set weight to 0 for axion(s) or photons with negative z momenta  
  if ((p4_gamma1[3] < 0) | (p4_gamma2[3] < 0) | (p4a[3] < 0)):
    event["event_weight"] = 0
    
  # set weight to 0 if beyond detector size
  # vertex should be automatically satisified given these requirements ONLY for no displacement
  if ((hit_gamma1[0]>l_x) | (hit_gamma1[1]>l_y) | (hit_gamma2[0]>l_x) | (hit_gamma2[1]>l_y) | (hit_gamma1[0]<-l_x) | (hit_gamma1[1]<-l_y) | (hit_gamma2[0]<-l_x) | (hit_gamma2[1]<-l_y) ):
    event["event_weight"] = 0
  
  # vertex needs to be checked explicitly in case displ_y!=0
  # might also check for ship truncated fustus
  
  # comment out l_x in case 
  lim_x = l_x
  if ship:
    lim_x = np.abs(0.6+1.35*(decay_position[2]-32)/50)
  if ((decay_position[0]>lim_x) | (decay_position[1]>l_y)  | (decay_position[0]<-lim_x) | (decay_position[1]<-l_y)  ):
    event["event_weight"] = 0

  
  # energy check  
  if ((p4_gamma1[0] < E_min) | (p4_gamma2[0] < E_min)  ):
    event["event_weight"] = 0
  
  # photon separation check  
  if ( np.sqrt((hit_gamma1[0]-hit_gamma2[0])**2+(hit_gamma1[1]-hit_gamma2[1])**2) < dr_gg  ):
    event["event_weight"] = 0
    
  return event

def random_event(m_alp, ctau_alp, z_min, z_max, z_cal, equal_weights, l_x, l_y, dr_gg, E_min): # Generates a random event for a randomly drawn ALP mass and decay length 
                                                                                                           # (assuming log priors with boundaries as specified)
                                                                                                                                                                                                                    
  reject = equal_weights

  if not(reject):
    b_meson = random.choice(b_mesons)
    p4a = b_meson_decay(b_meson, m_alp)
    event = alp_decay(p4a, m_alp, ctau_alp, z_min, z_max, z_cal, l_x, l_y, dr_gg, E_min)
          
  else: 
    while(reject):  # For equal_weights = True the probability to accept an event is equal to its weight, such that all accepted events have equal weight
      b_meson = random.choice(b_mesons)
      p4a = b_meson_decay(b_meson, m_alp)
      event = alp_decay(p4a, m_alp, ctau_alp, z_min, z_max, z_cal, l_x, l_y, dr_gg, E_min)
      acceptance = random.uniform(0,1)
      if acceptance < event["event_weight"]:
        event["event_weight"] = 1 
        reject = False
      
  return event

# n_obs is how many events seen per parameter points, n_events is the sample size
def generate_events(n_events, n_obs, m_alp_min, m_alp_max, ctau_min, ctau_max, z_min, z_max, z_cal, equal_weights, l_x, l_y, dr_gg, E_min, tau_mass): # Generates n_events events
  if m_alp_max > mB0 - mK0: # error in case you try to generate events which are unphysical
    raise ValueError('You are generating events with too massive ALPs: cannot be produced in this decay.\n  Lower m_alp_max to be smaller than {}'.format(mB0 - mK0))
    
  event_list = []
  while len(event_list) < n_events:
    m_alp = np.exp(random.uniform(np.log(m_alp_min),np.log(m_alp_max)))
    if tau_mass:
      ctau_alp = np.exp(np.log(m_alp)+random.uniform(np.log(ctau_min), np.log(ctau_max))) 
    else:
      ctau_alp = np.exp(random.uniform(np.log(ctau_min), np.log(ctau_max))) 
    event_obs = {
    "m_alp": m_alp,
    "ctau_alp": ctau_alp}
    iOBS =0
    while iOBS < n_obs:

        new_event = random_event(m_alp, ctau_alp, z_min, z_max, z_cal, equal_weights, l_x, l_y, dr_gg, E_min)
        if new_event["event_weight"] > 0: # reject invalid events
            for k_old in ["gamma_1_calo_hit", "gamma_1_4momentum", "gamma_2_calo_hit", "gamma_2_4momentum", "decay_position", "event_weight"]: # new keys are assigned to new events
                new_event[k_old+"_"+str(iOBS)] = new_event.pop(k_old)
            event_obs.update(new_event) # malp and talp values are overwritten, but they are the same as the n_obs events are generated for the same parameter values
            iOBS+=1
    event_list.append(event_obs)
  return event_list



# generate hepmc3 does not take several events per param point
# in case of more events, it will just use the first one (this can be changed by changing i_obs)
def generate_hepmc3(event_list): # Prints all events in HepMC3 format
  # by default  

  iOBS = 0 
  ALPPDG = 10001
  print("HepMC::Version 3.0.0")
  print("HepMC::Asciiv3-START_EVENT_LISTING")
  for i in range(len(event_list)):
    event = event_list[i]
    ALPmass = event["m_alp"]
    p4gamma1 = event["gamma_1_4momentum"+"_"+str(iOBS)]
    p4gamma2 = event["gamma_2_4momentum"+"_"+str(iOBS)]
    p4alp = p4gamma1 + p4gamma2
    vertex = event["decay_position"+"_"+str(iOBS)]
    print("E {} 1 3".format(i+1))
    print("U GEV MM")
    print("W {}".format(event["event_weight"+"_"+str(iOBS)]))
    print("P 1 0 {} {} {} {} {} {} 2".format(ALPPDG, p4alp[1], p4alp[2], p4alp[3], p4alp[0], ALPmass))
    print("V -1 0 [1] @ {} {} {} 0.0".format(vertex[0], vertex[1], vertex[2]))
    print("P 2 1 22 {} {} {} {} 0 1".format(p4gamma1[1], p4gamma1[2], p4gamma1[3], p4gamma1[0]))
    print("P 3 1 22 {} {} {} {} 0 1".format(p4gamma2[1], p4gamma2[2], p4gamma2[3], p4gamma2[0]))

  return
  
 
 
 # these will be directly in the notebook/generation script

# events = generate_events(n_events = 100, n_obs = 3, m_alp_min = 0.1, m_alp_max=1, ctau_min=0.1, ctau_max=10, z_min=1, z_max=100, z_cal=110, equal_weights = True, l_x=5, l_y=5, displ_y=0, dr_gg=0.1, E_min=1)
# generate_hepmc3(events)


