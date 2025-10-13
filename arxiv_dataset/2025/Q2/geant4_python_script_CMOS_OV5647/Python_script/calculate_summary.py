import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import os

def read_numpy_data(file):
    """Reads the CSV file and extracts the required columns."""
    data = np.loadtxt(file, delimiter="\t", skiprows=1, dtype=str)

    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    edep = data[:, 2].astype(float)
    ekinpre = data[:, 4].astype(float)
    ekin = data[:, 5].astype(float)
    eventID = data[:, 7].astype(int)
    parentID = data[:, 9].astype(int)
    trackID = data[:, 10].astype(int)

    # Filter rows where Edep > 0
    mask = edep > 0
    return x[mask], y[mask], edep[mask], ekinpre[mask], ekin[mask], eventID[mask], parentID[mask], trackID[mask]

def classify_and_sort_data(x, y, edep, ekinpre, ekin, eventID, parentID, trackID):
    """Classifies data by eventID, parentID and trackID and sorts by Ekinpre (highest to lowest)."""

    data = np.array(list(zip(x, y, edep, ekinpre, ekin, eventID, parentID, trackID)),
                   dtype=[('x', 'i4'), ('y', 'i4'), ('Edep', 'f8'), ('Ekinpre', 'f8'), ('Ekin', 'f8'),
                          ('eventID', 'i4'), ('Parent_ID', 'i4'), ('track_ID', 'i4')])

    sorted_data = np.sort(data, order=['eventID', 'Parent_ID', 'track_ID'])

    sorted_indices = np.lexsort((-sorted_data['Ekinpre'],
                                sorted_data['track_ID'],
                                sorted_data['Parent_ID'],
                                sorted_data['eventID']))
    sorted_data = sorted_data[sorted_indices]

    return sorted_data

def calculate_energy_difference(sorted_data):
    """Calculates energy difference and adds it as a column."""
    n_rows = len(sorted_data)
    forward_energy_difference = np.zeros(n_rows)
    backward_energy_difference = np.zeros(n_rows)

    for i in range(n_rows - 1):
        if (sorted_data[i]['eventID'] == sorted_data[i+1]['eventID'] and
            sorted_data[i]['Parent_ID'] == sorted_data[i+1]['Parent_ID'] and
            sorted_data[i]['track_ID'] == sorted_data[i+1]['track_ID']):
            forward_energy_difference[i] = sorted_data[i+1]['Ekinpre'] - sorted_data[i]['Ekin']

    for i in range(1, n_rows):
        if (sorted_data[i]['eventID'] == sorted_data[i-1]['eventID'] and
            sorted_data[i]['Parent_ID'] == sorted_data[i-1]['Parent_ID'] and
            sorted_data[i]['track_ID'] == sorted_data[i-1]['track_ID']):
            backward_energy_difference[i] = sorted_data[i]['Ekinpre'] - sorted_data[i-1]['Ekin']

    data_with_difference = np.zeros(n_rows, dtype=sorted_data.dtype.descr + [('dE_forward', 'f8'), ('dE_backward', 'f8'), ('group', 'i4')])
    for name in sorted_data.dtype.names:
        data_with_difference[name] = sorted_data[name]
    data_with_difference['dE_forward'] = forward_energy_difference
    data_with_difference['dE_backward'] = backward_energy_difference

    group = 0
    data_with_difference['group'][0] = group

    for i in range(1, n_rows):
        if (data_with_difference[i]['dE_backward'] != 0 or
            data_with_difference[i]['eventID'] != data_with_difference[i-1]['eventID'] or
            data_with_difference[i]['Parent_ID'] != data_with_difference[i-1]['Parent_ID'] or
            data_with_difference[i]['track_ID'] != data_with_difference[i-1]['track_ID']):
            group += 1
        data_with_difference['group'][i] = group

    return data_with_difference

def calculate_group_summary(data_with_difference, parent_id_filter=None):
    # Usage:  parent_id_filter=1 → Only Parent_ID == 1
    #         parent_id_filter=">1" → Only Parent_ID > 1
    #         parent_id_filter="<5" → Only Parent_ID < 5
    #         parent_id_filter=None → No filter applied

    """Calculates the maximum Ekinpre, sum of Edep and number of pixels per group."""
    # Apply filter if defined
    if parent_id_filter is not None:
        if isinstance(parent_id_filter, str):
            operator = parent_id_filter[0]
            value = int(parent_id_filter[1:])
            if operator == '>':
                data_with_difference = data_with_difference[data_with_difference['Parent_ID'] > value]
            elif operator == '<':
                data_with_difference = data_with_difference[data_with_difference['Parent_ID'] < value]
            elif operator == '=':
                data_with_difference = data_with_difference[data_with_difference['Parent_ID'] == value]
        else:
            data_with_difference = data_with_difference[data_with_difference['Parent_ID'] == parent_id_filter]

    unique_groups = np.unique(data_with_difference['group'])
    summary = []

    for group in unique_groups:
        group_data = data_with_difference[data_with_difference['group'] == group]
        ekinpre_max = np.max(group_data['Ekinpre'])
        edep_sum = np.sum(group_data['Edep'])
        edep_max = np.max(group_data['Edep'])
        num_pixels = len(group_data)
        eventID = group_data['eventID'][0]
        parentID = group_data['Parent_ID'][0]
        trackID = group_data['track_ID'][0]
        summary.append((ekinpre_max, edep_sum, edep_max, eventID, parentID, trackID, group, num_pixels))

    return np.array(summary, dtype=[('Ekinpre_max', 'f8'), ('Edep_sum', 'f8'), ('Edep_max', 'f8'), ('eventID', 'i4'), ('Parent_ID', 'i4'), ('track_ID', 'i4'), ('num_cluster', 'i4'), ('cluster_size', 'i4')])

def save_summary_cluster(group_summary, filename_base, formats=["txt", "csv", "npy", "npz"]):
    """Saves results in specified formats: TXT, CSV, NPY, NPZ."""

    if "txt" in formats:
        np.savetxt(f"{filename_base}.txt", group_summary, fmt='%.6f\t%.6f\t%.6f\t%d\t%d\t%d\t%d\t%d',
                  header='Ekinpre_max(MeV)\tEdep_sum(keV)\tEdep_max(keV)\teventID\tParent_ID\ttrack_ID\tnum_cluster\tcluster_size',
                  comments='')
    if "csv" in formats:
        np.savetxt(f"{filename_base}.csv", group_summary, delimiter=',', fmt='%.6f,%.6f,%.6f,%d,%d,%d,%d,%d',
                  header='Ekinpre_max(MeV),Edep_sum(keV),Edep_max(keV),eventID,Parent_ID,track_ID,num_cluster,cluster_size',
                  comments='')
    if "npy" in formats:
        np.save(f"{filename_base}.npy", group_summary)
    if "npz" in formats:
        np.savez_compressed(f"{filename_base}.npz", group_summary)

    print(f"Summary cluster files saved in formats: {', '.join(formats)}")

def save_energy_diff(data_with_difference, filename_base, formats=["txt", "csv", "npy", "npz"]):
    """Saves results in specified formats: TXT, CSV, NPY, NPZ."""

    if "txt" in formats:
        np.savetxt(f"{filename_base}.txt", data_with_difference, fmt='%d\t%d\t%.6f\t%.6f\t%.6f\t%d\t%d\t%d\t%.6f\t%.6f\t%d',
                  header='x\ty\tEdep(keV)\tEkinpre(MeV)\tEkin(MeV)\teventID\tParent_ID\ttrack_ID\tdE_forward(MeV)\tdE_backward(MeV)\tgroup',
                  comments='')
    if "csv" in formats:
        np.savetxt(f"{filename_base}.csv", data_with_difference, delimiter=',', fmt='%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%d',
                  header='x,y,Edep(keV),Ekinpre(MeV),Ekin(MeV),eventID,Parent_ID,track_ID,dE_forward(MeV),dE_backward(MeV),group',
                  comments='')
    if "npy" in formats:
        np.save(f"{filename_base}.npy", data_with_difference)
    if "npz" in formats:
        np.savez_compressed(f"{filename_base}.npz", data_with_difference)

    print(f"Energy difference files saved in formats: {', '.join(formats)}")

# Code usage
if __name__ == "__main__":
    start_time = time.time()

    parent_id_filter = None
    #parent_id_filter = 1

    source1 = 'Sr90'
    source2 = 'Cs137'

    nfrm = 1

    if source1 == 'Sr90':
        winback1 = '254'
        date1 = '2023_Oct_24_23h'
        ev1 = 1514000 // nfrm
        tim1 = ev1 / 3028
        evt1 = winback1 + '_ev' + str(ev1)
        z_count = 1

    if source2 == 'Cs137':
        winback2 = '635'
        date2 = '2023_Oct_23_23h'
        ev2 = 3811000 // nfrm
        tim2 = ev2 / 7622
        evt2 = winback2 + '_ev' + str(ev2)
        z_count = 1

    level_z = list(range(z_count))

    path_main = 'C:/dat_2025/emit/'
    path_name1 = f'data_{source1}_pixl_thickZ_2um_{z_count}level_{winback1}_100um_ev{ev1}_emit/'
    path_name2 = f'data_{source2}_pixl_thickZ_2um_{z_count}level_{winback2}_100um_ev{ev2}_emit/'

    if parent_id_filter == 1:
        dirfile = f'{path_main}clasfy_dat_00prim_sim_Edep_Ekin_{source1}_{source2}_{nfrm}f_{evt1}_{evt2}'
    if parent_id_filter is None:
        dirfile = f'{path_main}clasfy_dat_00all_sim_Edep_Ekin_{source1}_{source2}_{nfrm}f_{evt1}_{evt2}'

    unit = 'keV'
    if unit == 'keV':
        U_eV = 1000
    if unit == 'MeV':
        U_eV = 1000000

    try:
        os.makedirs(dirfile)
    except FileExistsError:
        pass

    line_width = 4
    font_size = 28

    particle_list = ['electron']  # 'gamma', 'particle'

    edep_condition = '_dif0'

    for particle_name in particle_list:
        for z in level_z:
            path_1 = f'data_{source1}_pixl_thickZ_2um_distSD_{z}mm_{z_count}level_csv_tar_bz/'
            path_2 = f'data_{source2}_pixl_thickZ_2um_distSD_{z}mm_{z_count}level_csv_tar_bz/'

            if particle_name in ['particle', 'electron', 'anti_nu_e', 'gamma']:
                particle_type = f'_edep{edep_condition}_all_{particle_name}_arrived_detector'

            for n in range(nfrm):
                print(n, particle_name)
                file1 = f'{path_main}{path_name1}{path_1}pixel{n}_SD{z}mm_{source1}{particle_type}.csv'
                file2 = f'{path_main}{path_name2}{path_2}pixel{n}_SD{z}mm_{source2}{particle_type}.csv'

                dirsave_plt_sim1 = f'classify_dat_{source1}_{nfrm}f{particle_type}'
                try:
                    os.makedirs(f'{dirfile}/{dirsave_plt_sim1}')
                except FileExistsError:
                    pass

                dirsave_plt_sim2 = f'classify_dat_{source2}_{nfrm}f{particle_type}'
                try:
                    os.makedirs(f'{dirfile}/{dirsave_plt_sim2}')
                except FileExistsError:
                    pass

                x1, y1, edep1, ekinpre1, ekin1, eventID1, parentID1, trackID1 = read_numpy_data(file1)
                sorted_data1 = classify_and_sort_data(x1, y1, edep1, ekinpre1, ekin1, eventID1, parentID1, trackID1)

                data_diff1 = calculate_energy_difference(sorted_data1)
                filesave_dat_dif1 = f'{dirfile}/{dirsave_plt_sim1}/dat_dif_{source1}_{particle_name}_ev{ev1}_z{str(z).zfill(3)}'
                save_energy_diff(data_diff1, filesave_dat_dif1, formats=["txt", "csv"])

                summary_cluster1 = calculate_group_summary(data_diff1, parent_id_filter)
                filesave_summary_clst1 = f'{dirfile}/{dirsave_plt_sim1}/summary_clst_{source1}_{particle_name}_ev{ev1}_z{str(z).zfill(3)}'
                save_summary_cluster(summary_cluster1, filesave_summary_clst1, formats=["txt", "csv", "npy", "npz"])

                x2, y2, edep2, ekinpre2, ekin2, eventID2, parentID2, trackID2 = read_numpy_data(file2)
                sorted_data2 = classify_and_sort_data(x2, y2, edep2, ekinpre2, ekin2, eventID2, parentID2, trackID2)

                data_diff2 = calculate_energy_difference(sorted_data2)
                filesave_dat_dif2 = f'{dirfile}/{dirsave_plt_sim2}/dat_dif_{source2}_{particle_name}_ev{ev2}_z{str(z).zfill(3)}'
                save_energy_diff(data_diff2, filesave_dat_dif2, formats=["txt", "csv"])

                summary_cluster2 = calculate_group_summary(data_diff2, parent_id_filter)
                filesave_summary_clst2 = f'{dirfile}/{dirsave_plt_sim2}/summary_clst_{source2}_{particle_name}_ev{ev2}_z{str(z).zfill(3)}'
                save_summary_cluster(summary_cluster2, filesave_summary_clst2, formats=["txt", "csv", "npy", "npz"])

                ################################################################################
                pair_eh = 3.6  # eV
                emax = 4300  # well capacity 
                emin = 5
                
                unit = 'keV'
                if unit == 'keV':
                    U_eV = 1000
                if unit == 'MeV':
                    U_eV = 1000000

                adc_min_limit = 101
                adc_max_limit = 1023
                min_cam = ((adc_min_limit-1)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                sat_cam = (1023*(emax-emin)/1023.0 + emin)*pair_eh/U_eV

                ###_Edep_vs_Ekin log #############################################################################
                ekinpre_keV = summary_cluster1['Ekinpre_max'] * 1000
                edep_keV = summary_cluster1['Edep_sum']

                font_size = 28
                fig, ax = plt.subplots(figsize=(15, 9.5))
                ax.tick_params(axis='both', labelsize=font_size)

                ax.axhline(y=sat_cam, color='k', linestyle="--", linewidth=2.5)
                ax.axhline(y=min_cam, color='k', linestyle="--", linewidth=2.5)

                bins_x = np.logspace(np.log10(min(ekinpre_keV)), np.log10(max(ekinpre_keV)), 50)
                bins_y = np.logspace(np.log10(min(edep_keV)), np.log10(max(edep_keV)), 50)

                hist, xedges, yedges = np.histogram2d(ekinpre_keV, edep_keV, bins=[bins_x, bins_y])
                hist = hist / hist.sum()  # Normalize histogram

                im = ax.pcolormesh(xedges, yedges, hist.T, norm=matplotlib.colors.LogNorm(vmin=0.7e-5, vmax=2.5e-2), cmap=plt.cm.jet)
                cbar = plt.colorbar(im, label='Normalized clusters')
                cbar.ax.tick_params(labelsize=font_size - 4)
                cbar.ax.set_ylabel('Normalized clusters', fontsize=font_size - 2)

                ax.text(1, sat_cam, '1023 ADC', ha='left', va='bottom', fontsize=font_size - 4)
                ax.text(1, min_cam, '100 ADC', ha='left', va='bottom', fontsize=font_size - 4)

                ax.set_xscale('log')
                ax.set_xlim(1, max(ekinpre_keV))
                ax.set_yscale('log')
                ax.set_ylim(1e-2, 2e2)
                ax.set_xlabel('$E^{e-}_{kin} $(keV)', fontsize=font_size-2)
                ax.set_ylabel('$E_{dep} $(keV)', fontsize=font_size-2)
                ax.set_title(source1, fontsize=font_size-4)
                plt.tight_layout()
                plt.savefig(f'{dirfile}/hist2d_normlog_{source1}_Edep_vs_Ekin{particle_type}.png', dpi=150)
                plt.savefig(f'{dirfile}/hist2d_normlog_{source1}_Edep_vs_Ekin{particle_type}.pdf', dpi=150)

                plt.show()

                ekinpre_keV = summary_cluster2['Ekinpre_max'] * 1000
                edep_keV = summary_cluster2['Edep_sum']

                font_size = 28
                fig, ax = plt.subplots(figsize=(15, 9.5))
                ax.tick_params(axis='both', labelsize=font_size)

                ax.axhline(y=sat_cam, color='k', linestyle="--", linewidth=2.5)
                ax.axhline(y=min_cam, color='k', linestyle="--", linewidth=2.5)

                bins_x = np.logspace(np.log10(min(ekinpre_keV)), np.log10(max(ekinpre_keV)), 50)
                bins_y = np.logspace(np.log10(min(edep_keV)), np.log10(max(edep_keV)), 50)

                hist, xedges, yedges = np.histogram2d(ekinpre_keV, edep_keV, bins=[bins_x, bins_y])
                hist = hist / hist.sum()  # Normalize histogram

                im = ax.pcolormesh(xedges, yedges, hist.T, norm=matplotlib.colors.LogNorm(vmin=0.7e-5, vmax=2.5e-2), cmap=plt.cm.jet)
                cbar = plt.colorbar(im, label='Normalized clusters')
                cbar.ax.tick_params(labelsize=font_size - 4)
                cbar.ax.set_ylabel('Normalized clusters', fontsize=font_size - 2)

                ax.text(1, sat_cam, '1023 ADC', ha='left', va='bottom', fontsize=font_size - 4)
                ax.text(1, min_cam, '100 ADC', ha='left', va='bottom', fontsize=font_size - 4)

                ax.set_xscale('log')
                ax.set_xlim(1, max(ekinpre_keV))
                ax.set_yscale('log')
                ax.set_ylim(1e-2, 2e2)
                ax.set_xlabel('$E^{e-}_{kin} $(keV)', fontsize=font_size - 2)
                ax.set_ylabel('$E_{dep} $(keV)', fontsize=font_size - 2)
                ax.set_title(source2, fontsize=font_size - 4)
                plt.tight_layout()
                plt.savefig(f'{dirfile}/hist2d_normlog_{source2}_Edep_vs_Ekin{particle_type}.png', dpi=150)
                plt.savefig(f'{dirfile}/hist2d_normlog_{source2}_Edep_vs_Ekin{particle_type}.pdf', dpi=150)

                plt.show()

                ###_Ekin_vs_cltrsiz lin #############################################################################
                ekinpre_keV = summary_cluster1['Ekinpre_max'] * 1000
                cltrsiz = summary_cluster1['cluster_size']

                font_size = 28
                fig, ax = plt.subplots(figsize=(15, 9.5))
                ax.tick_params(axis='both', labelsize=font_size)

                bins_x = max(cltrsiz)-1
                bins_y = max(cltrsiz)-1

                hist, xedges, yedges = np.histogram2d(ekinpre_keV, cltrsiz, bins=[bins_x, bins_y])
                hist = hist / hist.sum()  # Normalize histogram
                print(yedges)

                im = ax.pcolormesh(xedges, yedges, hist.T, norm=matplotlib.colors.LogNorm(vmin=0.6e-5, vmax=4e-2), cmap=plt.cm.jet)
                cbar = plt.colorbar(im, label='Normalized clusters')
                cbar.ax.tick_params(labelsize=font_size - 4)
                cbar.ax.set_ylabel('Normalized clusters', fontsize=font_size - 2)

                ax.set_xlim(1e-2, max(ekinpre_keV))
                ax.set_ylim(1, 40)
                ax.set_xlabel('$E^{e-}_{kin} $(keV)', fontsize=font_size-2)
                ax.set_ylabel('Cluster Size', fontsize=font_size-2)
                ax.set_title(source1, fontsize=font_size-4)
                plt.tight_layout()
                plt.savefig(f'{dirfile}/hist2d_line_norm_{source1}_cltrsiz_vs_Ekin{particle_type}.png', dpi=150)
                plt.savefig(f'{dirfile}/hist2d_line_norm_{source1}_cltrsiz_vs_Ekin{particle_type}.pdf', dpi=150)

                plt.show()

                ################################################################################
                ekinpre_keV = summary_cluster2['Ekinpre_max'] * 1000
                cltrsiz = summary_cluster2['cluster_size']

                font_size = 28
                fig, ax = plt.subplots(figsize=(15, 9.5))
                ax.tick_params(axis='both', labelsize=font_size)

                bins_x = max(cltrsiz)-1
                bins_y = max(cltrsiz)-1

                hist, xedges, yedges = np.histogram2d(ekinpre_keV, cltrsiz, bins=[bins_x, bins_y])
                hist = hist / hist.sum()  # Normalize histogram
                print(yedges)

                im = ax.pcolormesh(xedges, yedges, hist.T, norm=matplotlib.colors.LogNorm(vmin=0.6e-5, vmax=4e-2), cmap=plt.cm.jet)
                cbar = plt.colorbar(im, label='Normalized clusters')
                cbar.ax.tick_params(labelsize=font_size - 4)
                cbar.ax.set_ylabel('Normalized clusters', fontsize=font_size - 2)

                ax.set_xlim(1e-2, max(ekinpre_keV))
                ax.set_ylim(1, 40)
                ax.set_xlabel('$E^{e-}_{kin} $ (keV)', fontsize=font_size-2)
                ax.set_ylabel('Cluster Size', fontsize=font_size-2)
                ax.set_title(source2, fontsize=font_size-4)
                plt.tight_layout()
                plt.savefig(f'{dirfile}/hist2d_line_norm_{source2}_cltrsiz_vs_Ekin{particle_type}.png', dpi=150)
                plt.savefig(f'{dirfile}/hist2d_line_norm_{source2}_cltrsiz_vs_Ekin{particle_type}.pdf', dpi=150)

                plt.show()

                ###_Edep_max_vs_Ekin log #############################################################################
                ekinpre_keV = summary_cluster1['Ekinpre_max'] * 1000
                edep_max_keV = summary_cluster1['Edep_max']

                font_size = 28
                fig, ax = plt.subplots(figsize=(15, 9.5))
                ax.tick_params(axis='both', labelsize=font_size)

                ax.axhline(y=sat_cam, color='k', linestyle="--", linewidth=2.5)
                ax.axhline(y=min_cam, color='k', linestyle="--", linewidth=2.5)

                bins_x = np.logspace(np.log10(min(ekinpre_keV)), np.log10(max(ekinpre_keV)), 50)
                bins_y = np.logspace(np.log10(min(edep_max_keV)), np.log10(max(edep_max_keV)), 50)

                hist, xedges, yedges = np.histogram2d(ekinpre_keV, edep_max_keV, bins=[bins_x, bins_y])
                hist = hist / hist.sum()  # Normalize histogram

                im = ax.pcolormesh(xedges, yedges, hist.T, norm=matplotlib.colors.LogNorm(vmin=0.7e-5, vmax=2.5e-2), cmap=plt.cm.jet)
                cbar = plt.colorbar(im, label='Normalized clusters')
                cbar.ax.tick_params(labelsize=font_size - 4)
                cbar.ax.set_ylabel('Normalized clusters', fontsize=font_size - 2)

                ax.text(0.5, sat_cam, '1023 ADC', ha='left', va='bottom', fontsize=font_size - 4)
                ax.text(0.5, min_cam, '100 ADC', ha='left', va='bottom', fontsize=font_size - 4)

                ax.set_xscale('log')
                ax.set_xlim(0.5, max(ekinpre_keV))
                ax.set_yscale('log')
                ax.set_ylim(1e-2, 2e2)
                ax.set_xlabel('$E^{e-}_{kin} $(keV)', fontsize=font_size-2)
                ax.set_ylabel('max $E_{dep} $(keV)', fontsize=font_size-2)
                ax.set_title(source1, fontsize=font_size-4)
                plt.tight_layout()
                plt.savefig(f'{dirfile}/hist2d_normlog_{source1}_Edep_max_vs_Ekin{particle_type}.png', dpi=150)
                plt.savefig(f'{dirfile}/hist2d_normlog_{source1}_Edep_max_vs_Ekin{particle_type}.pdf', dpi=150)

                plt.show()

                ekinpre_keV = summary_cluster2['Ekinpre_max'] * 1000
                edep_max_keV = summary_cluster2['Edep_max']

                font_size = 28
                fig, ax = plt.subplots(figsize=(15, 9.5))
                ax.tick_params(axis='both', labelsize=font_size)

                ax.axhline(y=sat_cam, color='k', linestyle="--", linewidth=2.5)
                ax.axhline(y=min_cam, color='k', linestyle="--", linewidth=2.5)

                bins_x = np.logspace(np.log10(min(ekinpre_keV)), np.log10(max(ekinpre_keV)), 50)
                bins_y = np.logspace(np.log10(min(edep_max_keV)), np.log10(max(edep_max_keV)), 50)

                hist, xedges, yedges = np.histogram2d(ekinpre_keV, edep_max_keV, bins=[bins_x, bins_y])
                hist = hist / hist.sum()  # Normalize histogram

                im = ax.pcolormesh(xedges, yedges, hist.T, norm=matplotlib.colors.LogNorm(vmin=0.7e-5, vmax=2.5e-2), cmap=plt.cm.jet)
                cbar = plt.colorbar(im, label='Normalized clusters')
                cbar.ax.tick_params(labelsize=font_size - 4)
                cbar.ax.set_ylabel('Normalized clusters', fontsize=font_size - 2)

                ax.text(0.5, sat_cam, '1023 ADC', ha='left', va='bottom', fontsize=font_size - 4)
                ax.text(0.5, min_cam, '100 ADC', ha='left', va='bottom', fontsize=font_size - 4)

                ax.set_xscale('log')
                ax.set_xlim(0.5, max(ekinpre_keV))
                ax.set_yscale('log')
                ax.set_ylim(1e-2, 2e2)
                ax.set_xlabel('$E^{e-}_{kin} $(keV)', fontsize=font_size-2)
                ax.set_ylabel('max $E_{dep} $(keV)', fontsize=font_size-2)
                ax.set_title(source1, fontsize=font_size-4)
                plt.tight_layout()
                plt.savefig(f'{dirfile}/hist2d_normlog_{source2}_Edep_max_vs_Ekin{particle_type}.png', dpi=150)
                plt.savefig(f'{dirfile}/hist2d_normlog_{source2}_Edep_max_vs_Ekin{particle_type}.pdf', dpi=150)                                                      

                plt.show()
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

