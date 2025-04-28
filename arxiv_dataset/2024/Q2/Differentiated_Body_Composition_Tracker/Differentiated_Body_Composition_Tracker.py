import numpy as np
import time
import sys
from astropy import units as u

start_time = time.time() # Timer to see how long running the code takes

# Units
kg_per_m_cubed = u.kg/u.m**3
Msun_per_au_cubed = u.Msun/u.au**3

# Densities
core_density = ((7874.0*kg_per_m_cubed).to(Msun_per_au_cubed)) # Density of iron
mantle_density = ((3000.0*kg_per_m_cubed).to(Msun_per_au_cubed)) # Density of mantle material

def organize_compositions(init_compositions):
    """Data organizing function
    
    Takes in raw string data extracted from composition input file
    Creates a list filled with properties of each particle
    The properties include hash, mass, and CMF
    
    Parameters:
    init_compositions (list) -- raw particle data from input file

    Returns:
    compositions (list) -- nested list with properly formatted particle data
    """
    compositions = []
    for particle in init_compositions: 
        particle_data = [] 
        for i in range(len(particle)): 
            if i == 0: 
                particle_data.append(int(particle[i])) # The particle's REBOUND hash
            elif i < 3: 
                particle_data.append(float(particle[i])) # Adds its mass than CMF
        compositions.append(particle_data) 
    return(compositions)

def calc_core_radius(mass, core_frac, core_density):
    """Returns the radius of a differentiated object's spherical, uniform core"""
    core_mass = mass*core_frac
    core_radius = ((3*core_mass)/(4*np.pi*core_density))**(1/3)
    return(core_radius)

def calc_radius(mass, core_frac, mantle_density, core_radius):
    """Returns the outer radius of a differentiated object with a known core radius"""
    mantle_frac = 1-core_frac
    mantle_mass = mass*mantle_frac
    radius = (core_radius**3+((3*mantle_mass)/(4*np.pi*mantle_density)))**(1/3)
    return(radius)

def calc_ejecta_core_frac(collision_type, impact_parameter, min_ejecta_cmf, max_ejecta_cmf, targ_radius, targ_core_radius, proj_radius, proj_core_radius):
    """Mantle Stripping Model
    
    Calculates CMF in collisional ejecta based on impact parameters
    See Ferich et al. (in prep) for details of model
    
    Parameters:
    collision_type (int) -- can be either accretive (2) or erosive (3 & 4)
    impact_parameter (float) -- B in model equations from Ferich et al. (in prep)
    min_ejecta_cmf (float) -- minimum possible CMF of impact ejecta
    max_ejecta_cmf (float) -- maximum possible CMF of impact ejecta
    targ_radius (float) -- outer radius of target
    targ_core_radius (float) -- radius of target's core
    proj_radius (float) -- outer radius of projectile
    proj_core_radius (float) -- radius of projectile's core
    
    Returns:
    ejecta_cmf (float) -- The ideal fraction of core material in the collision ejecta
    """
    ejecta_cmf = 0.0
    if collision_type == 2:
        if proj_core_radius != 0:
            slope = (max_ejecta_cmf-min_ejecta_cmf)/(-2*proj_core_radius)
            max_impact_parameter = targ_radius+proj_core_radius 
            min_impact_parameter = targ_radius-proj_core_radius 
            if impact_parameter >= max_impact_parameter: # Oblique collisions
                ejecta_cmf = min_ejecta_cmf 
            elif max_impact_parameter > impact_parameter > min_impact_parameter: 
                ejecta_cmf = min_ejecta_cmf+(slope*(impact_parameter-max_impact_parameter)) 
            elif impact_parameter <= min_impact_parameter: # Head-on collisions 
                ejecta_cmf = max_ejecta_cmf 
    elif collision_type == 3 or collision_type == 4:
        if targ_core_radius != 0:
            slope = (max_ejecta_cmf-min_ejecta_cmf)/(-2*targ_core_radius)
            max_impact_parameter = proj_radius+targ_core_radius
            min_impact_parameter = proj_radius-targ_core_radius 
            if impact_parameter >= max_impact_parameter:  # Oblique collisions
                ejecta_cmf += min_ejecta_cmf 
            elif max_impact_parameter > impact_parameter > min_impact_parameter: 
                ejecta_cmf += min_ejecta_cmf+(slope*(impact_parameter-max_impact_parameter)) 
            elif impact_parameter <= min_impact_parameter: # Head-on collisions
                ejecta_cmf += max_ejecta_cmf         
    return(ejecta_cmf)

def calc_ejecta_core_mantle_masses(targ_mass, lr_mass, obj_mass, obj_radius, obj_cmf, ejecta_cmf):
    """Calculates amount of core and mantle material in ejecta
    
    If the core or mantle of the stripped object doesn't have enough
    material to reach the ideal CMF for the collision ejecta, then
    this function will pull more material out of the other layer.
    Basically makes sure that the code doesn't put core or mantle material that
    the stripped object doesn't have into the ejecta.
    
    
    Parameters:
    targ_mass (float) -- Mass of the target
    lr_mass (float) -- Mass of the largest remnant
    obj_mass (float) -- Mass of the stripped object in collision
    obj_radius (float) -- Radius of the stripped object in collision
    obj_cmf (float) -- Core mass fraction of stripped object
    ejecta_cmf (float) -- the ideal CMF of the collision ejecta

    Returns:
    actual_ejecta_masses (list) -- non-ideal core and mantle masses of the ejecta
    """
    ejecta_mass = abs(targ_mass-lr_mass)
    ideal_ejecta_masses = [(ejecta_mass*ejecta_cmf),(ejecta_mass*(1-ejecta_cmf))] # Ideal masses of the ejecta's core and mantle
    obj_layer_masses = [obj_mass*obj_cmf, obj_mass*(1-obj_cmf)] # How much mass the object actually has in each layer
    actual_ejecta_masses = [] # Actual masses of the ejecta's core and mantle
    if obj_layer_masses[0] >= ideal_ejecta_masses[0]:
        actual_ejecta_masses.append(ideal_ejecta_masses[0])
    else:
        actual_ejecta_masses.append(obj_layer_masses[0])
        ideal_ejecta_masses[1] += (ideal_ejecta_masses[0]-obj_layer_masses[0])
    if obj_layer_masses[1] >= ideal_ejecta_masses[1]:
        actual_ejecta_masses.append(ideal_ejecta_masses[1])
    else:
        actual_ejecta_masses.append(obj_layer_masses[1])
        actual_ejecta_masses[0] += (ideal_ejecta_masses[1]-obj_layer_masses[1])
    return(actual_ejecta_masses)



def track_composition(collision_report_file, composition_input_file, ejection_file, min_core_collision_frac, max_core_collision_frac):
    """Main function
    
    Tracks how collisions change the CMFs of objects from REBOUND fragmentation sim
    Returns the final compositions of all remaining objects 
    Has 2 main sections for mergers and disruptive collisions (accretive and erosive)
    
    
    Parameters:
    collision_report_file (str) -- pathway to collision report file
    collision_input_file (str) -- pathway to the DBCT input file
    ejection_file (str) -- pathway to file that lists objects ejected from sim
    min_core_collision_frac (float) -- minimum possible CMF of impact ejecta
    max_core_collision_frac (float) -- maximum possible CMF of impact ejecta

    Returns:
    compositions (list) -- nested list with compositional data of final objects
    """
    # Extracts compositional data
    f = open(composition_input_file, 'r')
    raw_compositions = [line.split() for line in f.readlines()]
    compositions = organize_compositions(raw_compositions) # This list keeps track of all the particle data
    for obj in compositions:
        if obj[2] > 1.0 or obj[2] < 0.0:
            print ('ERROR: CMF does not have a realistic value')
            sys.exit(1) 
    f.close() 
    
    # Extracts collisional data  
    f = open(collision_report_file, 'r')
    collision_blocks = f.read().split("\n")
    collisions = [block for block in collision_blocks if len(block) > 0]
    f.close()
    
    destroyed_object_hashes = [] # Keeps track of objects that get destroyed in collisions
    
    ############### START OF MAIN LOOP ############### 
    # Iterates through every collision from sim
    # Determines the change in objects' compositions from each collision
    # Adds new fragments to the compositions list when necessary
    for i in range(len(collisions)): 
        collision = collisions[i].split() 
        time = float(collision[0]) 
        collision_type = int(collision[1]) 
        if collision_type == 0: # Elastic bounces do nothing so loop moves to next collision
            continue
        sim_impact_param = float(collision[2])
        target_hash = int(collision[3]) 
        largest_remnant_mass = float(collision[4]) # Mass of the target after collision
        target_sim_radius = float(collision[5])/5
        proj_hash = int(collision[6]) 
        proj_sim_radius = float(collision[7])/5
        no_frags = int((len(collision)-8)/2)
        frag_hashes = [int(collision[j*2+6]) for j in range(1,no_frags+1)]
        frag_masses = [float(collision[j*2+7]) for j in range(1,no_frags+1)]

        # Determines if the projectile collided with something not in the compositions list (Star, Jupiter, etc.)
        big_obj_collision_flag = 1
        for obj in compositions:
            if target_hash == obj[0]:
                big_obj_collision_flag += -1
                break
        if big_obj_collision_flag == 1:
            destroyed_object_hashes.append(proj_hash) # Projectile is always considered destroyed in this type of collision
            if no_frags != 0: # If fragments created, adds those to compositions list with CMF of 0.3
                for j in range(no_frags):
                     frag_data = [frag_hashes[j], frag_masses[j], 0.3]
                     compositions.append(frag_data) 
            continue # Moves to the next collision
        
        targ_idx = [idx for idx in range(len(compositions)) if int(compositions[idx][0])==target_hash][0] # Index of target in the compositions list
        proj_idx = [idx for idx in range(len(compositions)) if int(compositions[idx][0])==proj_hash][0] # Index of projectile in the compositions list
        target_mass = float(compositions[targ_idx][1]) 
        proj_mass = float(compositions[proj_idx][1]) 
        target_core_frac = compositions[targ_idx][2]
        proj_core_frac = compositions[proj_idx][2] 
        
        # This sequence estimates the core and outer radii of the target and projectile
        target_core_radius = calc_core_radius(target_mass, target_core_frac, core_density)
        target_radius = calc_radius(target_mass, target_core_frac, mantle_density, target_core_radius)
        proj_core_radius = calc_core_radius(proj_mass, proj_core_frac, core_density)
        proj_radius = calc_radius(proj_mass, proj_core_frac, mantle_density, proj_core_radius)
        sine_of_impact_angle = sim_impact_param/(target_sim_radius+proj_sim_radius)
        impact_parameter = (target_radius+proj_radius)*sine_of_impact_angle # New impact parameter with updated radii
        
                                               
    ############### PERFECT MERGER ############### 
        if collision_type == 1: #perfect merger
            compositions[targ_idx][2] = ((target_core_frac*target_mass)+(proj_core_frac*proj_mass))/largest_remnant_mass #changes the composition fraction for each specie in the target - basically weighted average of initial target compoisition and mass with the projectile composition and mass
 
    ############### DISRUPTIVE COLLISIONS ############### 
        elif collision_type == 2 or collision_type == 3 or collision_type == 4:
            ejecta_core_frac = calc_ejecta_core_frac(collision_type, impact_parameter, min_core_collision_frac, max_core_collision_frac, target_radius, target_core_radius, proj_radius, proj_core_radius)
            if collision_type == 2:
                ejecta_layer_masses = calc_ejecta_core_mantle_masses(target_mass, largest_remnant_mass, proj_mass, proj_radius, proj_core_frac, ejecta_core_frac)
                compositions[targ_idx][2] = ((target_mass*target_core_frac)+(ejecta_layer_masses[0]))/largest_remnant_mass # Changes target CMF to largest remnant CMF for accretion
            else:
                ejecta_layer_masses = calc_ejecta_core_mantle_masses(target_mass, largest_remnant_mass, target_mass, target_radius, target_core_frac, ejecta_core_frac)
                compositions[targ_idx][2] = ((target_mass*target_core_frac)-(ejecta_layer_masses[0]))/largest_remnant_mass # Changes target CMF to largest remnant CMF for erosion
                
            
            # Target CMF calculation tolerance
            if compositions[targ_idx][2] - 1.0 > 0.0 and compositions[targ_idx][2] - 1.0 < 1.0e-5: # If target CMF is just above 1.0
                compositions[targ_idx][2] = 1.0
            if compositions[targ_idx][2] < 0.0 and compositions[targ_idx][2] > -1.0e-5: # If target CMF is just below 0.0
                compositions[targ_idx][2] = 0.0
            
            total_core_mass = (target_core_frac*target_mass)+(proj_core_frac*proj_mass) 
            largest_remnant_core_mass = largest_remnant_mass*compositions[targ_idx][2] 
            total_frag_core_mass = total_core_mass - largest_remnant_core_mass 
            total_frag_mass = (target_mass+proj_mass) - largest_remnant_mass 
            frag_core_frac = total_frag_core_mass/total_frag_mass # All fragments get the same CMF
            
            # Fragment CMF calculation tolerance
            if frag_core_frac - 1.0 > 0.0 and frag_core_frac - 1.0 < 1.0e-5: # If fragment CMF is just above 1.0
                frag_core_frac += 1.0 - frag_core_frac
            if frag_core_frac < 0.0 and frag_core_frac > -1.0e-5: # If fragment CMF is just below 0.0
                frag_core_frac += 0.0 - frag_core_frac
            
            # Fragment CMF error
            if frag_core_frac < 0.0:
                print ('ERROR: Fragment CMF is negative at time: ', time)
                sys.exit(1)
            elif frag_core_frac > 1.0:
                print ('ERROR: Fragment CMF is greater than 1.0 at time: ', time)
                sys.exit(1)
                
            for j in range(no_frags):
                if frag_masses[j] < 0:
                    print ('ERROR: Fragment mass is negative at time: ', time)
                    sys.exit(1)
                else:
                    frag_data = [frag_hashes[j], frag_masses[j], frag_core_frac]
                    compositions.append(frag_data) # Fragment now added to the object tracking list
                
        compositions[targ_idx][1] = largest_remnant_mass # Changes target mass to largest remnant mass
        
        # Checks to see if there are any errors with the largest remnant's properties 
        for j in range(len(compositions[targ_idx])):
            if j == 1:
                if compositions[targ_idx][j] < 0.0: 
                    print ('ERROR: Largest remnant mass is negative at time: ', time)
                    sys.exit(1)
            elif j > 1:
                if compositions[targ_idx][j] < 0.0:
                    print ('ERROR: Largest remnant CMF is negative at time: ', time)
                    sys.exit(1)
                elif compositions[targ_idx][j] > 1.0:
                    print ('ERROR: Largest remnant CMF is greater than 1.0 at time: ', time)
                    sys.exit(1)
                    
                    
        # Checks to make sure the projectile isn't a second largest remnant before deletion
        for hsh in frag_hashes:
            if hsh != proj_hash and hsh == frag_hashes[-1]:
                destroyed_object_hashes.append(proj_hash) 
            elif hsh == proj_hash:
                break
            else:
                continue
        
        if collision_type == 1:  
            destroyed_object_hashes.append(proj_hash) # The projectile in a merger is always destroyed
    
    ############### END OF MAIN LOOP ###############
    
    # Removes destroyed objects from compositions list    
    for hsh in destroyed_object_hashes:
        for i in range(len(compositions)):
            if compositions[i][0] == hsh:
                compositions.pop(i)
                break
            else:
                continue
    
    # Removes ejected objects from compositions list
    f = open(ejection_file, 'r')
    ejections_raw = [line.split() for line in f.readlines()]
    ejections = [abs(int(ejections_raw[i][1])) for i in range(len(ejections_raw))]
    for hsh in ejections:
        for i in range(len(compositions)):
            if compositions[i][0] == hsh:
                compositions.pop(i)
                break
            else:
                continue
    f.close()
         
    return(compositions)


def write_output(compositions, composition_output_file):
    """Writes final objects and their propeties to output file"""
    f = open(composition_output_file, "w")
    for obj in compositions: 
        for i in range(len(obj)): 
                f.write(str(obj[i]) + ' ')
        f.write('\n')#go to a new line and to a new particle
    f.close()
 

# Input Files
collision_file = "DBCT_input/collision_report.txt"
comp_input_file = "DBCT_input/initial_compositions.txt"
ejec_file = "DBCT_input/ejections.txt"

# Output Files
comp_output_file = "DBCT_output/DBCT_output.txt"

# DBCT Model Parameters
min_ejecta_core_frac = 0.0
max_ejecta_core_frac = 1.0

# Runs DBCT and outputs file
write_output(track_composition(collision_file, comp_input_file, ejec_file, min_ejecta_core_frac, max_ejecta_core_frac), comp_output_file)


print("--- %s seconds ---" % (time.time() - start_time))