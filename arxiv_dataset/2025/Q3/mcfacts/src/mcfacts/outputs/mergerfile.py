"""BBH merger output utilities

Utilities for handling BBH merger outputs

Notes
-----
    'galaxy' : The galaxy number mcfacts was on when a merger was detected
    'bin_com' : The binary center of mass prior to merger
    'final_mass' : The final mass of the merged remnant
    'chi_eff' : The effective spin prior to merger
    'final_spin' : Spin of the remnant post-merger
    'spin_angle' : The spin angle is zero for now
    'mass_1' : The mass of the first component prior to merger
    'mass_2' : The mass of the second component prior to merger
    'a_1' : The spin of the first component prior to merger
    'a_2' : The spin of the second component prior to merger
    'theta_1' : The spin angle of the first component prior to merger
    'theta_2' : The spin angle of the second component prior to merger
    'gen1' : The merger generation of the first component
    'gen2' : The merger generation of the second component
    'time_merge' : The timestep of merger
    'chi_p' : The precessing spin component of the binary prior to merger
"""
import numpy as np

MERGER_FIELD_NAMES = [
             'galaxy',
             'bin_com', 'final_mass', 'chi_eff', 'final_spin',
             'spin_angle', 'mass_1', 'mass_2', 'spin1',
             'spin2', 'theta1', 'theta2', 'gen1',
             'gen2', 'time_merge', 'chi_p']
#dtype_rec = np.dtype( [(x, float) for x in names_rec])

def merged_bh(
    merged_bh_array,
    binary_bh_array,
    merger_indices,
    merger_id,
    chi_eff,
    final_mass,
    final_spin,
    n_mergers_so_far,
    chi_p,
    time_passed
    ):
    """Return an array with properties of merger

    Designed only to handle one event at a time, with hardcoded labels

    Parameters
    ----------
    merged_bh_array : numpy.ndarray (len(MERGER_FIELD_NAMES),bin_num_max)
        The output array for merger properties.
        This was passed by reference, and we modify it within this function
    binary_bh_array : numpy.ndarray (bin_properties_num, bin_num_max)
        Array containing binary bbhs
        (see mcfacts_sim.py -> binary_field_names)
    merger_indices : numpy.ndarray
        An array indicating merger indices for the current timestep
    merger_id : int
        id for the merger we are currently adding to the merged_bh array
    chi_eff : float
        effective spin of the binary
    final_mass : float
        final remnant mass of the merger product
    final_spin : float
        spin of the merger product
    n_mergers_so_far : int
        Total number of mergers for previous timesteps in this galaxy
    chi_p : float
        The precessing spin component of the binary prior to merger
    time_passed : float
        The time of the current timestep in the mcfacts loop
    """
    #Comment here
    #Center of mass, M_total, Chi_eff, a_tot, spin_angle, m1,m2,a1,a2,theta1,theta2,gen1,gen2,time of merger, chi_eff
    merged_bh_array[0,n_mergers_so_far + merger_id] = binary_bh_array[9,merger_indices[merger_id]]
    merged_bh_array[1,n_mergers_so_far + merger_id] = final_mass
    merged_bh_array[2,n_mergers_so_far + merger_id] = chi_eff
    merged_bh_array[3,n_mergers_so_far + merger_id] = final_spin
    merged_bh_array[4,n_mergers_so_far + merger_id] = 0.0
    merged_bh_array[5,n_mergers_so_far + merger_id] = binary_bh_array[2,merger_indices[merger_id]]
    merged_bh_array[6,n_mergers_so_far + merger_id] = binary_bh_array[3,merger_indices[merger_id]]
    merged_bh_array[7,n_mergers_so_far + merger_id] = binary_bh_array[4,merger_indices[merger_id]]
    merged_bh_array[8,n_mergers_so_far + merger_id] = binary_bh_array[5,merger_indices[merger_id]]
    merged_bh_array[9,n_mergers_so_far + merger_id] = binary_bh_array[6,merger_indices[merger_id]]
    merged_bh_array[10,n_mergers_so_far + merger_id] = binary_bh_array[7,merger_indices[merger_id]]
    merged_bh_array[11,n_mergers_so_far + merger_id] = binary_bh_array[14,merger_indices[merger_id]]
    merged_bh_array[12,n_mergers_so_far + merger_id] = binary_bh_array[15,merger_indices[merger_id]]
    merged_bh_array[13,n_mergers_so_far + merger_id] = time_passed
    merged_bh_array[14,n_mergers_so_far + merger_id] = chi_p
    # We don't need to return this anymore, because we are mutating the merger array
    # which was passed by reference.
    #return merged_bh_array[:,n_mergers_so_far + merger_id]
