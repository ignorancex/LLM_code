import os
import sys
import unyt
import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Tuple

# Make the register backend visible to the script
sys.path.append(
    os.path.abspath(
        os.path.join(
            '.',
            os.path.pardir,
        )
    )
)

from read import MacsisDataset
from register import Macsis
from utils import save_dict_to_hdf5, load_dict_from_hdf5

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

color_palette = iter(['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600', ])
macsis = Macsis()


class MassMatchedRedshiftSampler(object):
    
    def __init__(
        self, 
        redshift_index_list: List[int], 
        mass_log10_range: Tuple[float, float] = (13, 16), 
        nbins: int = 40
    ) -> None:
        
        assert redshift_index_list == sorted(redshift_index_list), "The redshift_index_list provided must be sorted."
        assert mass_log10_range[0] < mass_log10_range[1], "Minimum mass must be less than maximum mass."
        assert nbins > 2, "Number of bins must be 3 or larger."
        
        self.min_max_redshifts = [min(redshift_index_list), max(redshift_index_list)]
        self.redshift_index_list = redshift_index_list
        
        self.bins = np.linspace(*mass_log10_range, nbins)
        self.bin_centres = (self.bins[1:] + self.bins[:-1]) / 2
        self.bin_size = (self.bins.max() - self.bins.min()) / len(self.bins)      
        
        
        properties = load_dict_from_hdf5(f"{macsis.output_dir}/properties_022.hdf5")
        m500 = properties['m_500crit'] * 1E10
        m200 = properties['m_200crit'] * 1E10
        fgas = properties['hot_gas_fraction_500crit']
        macsis_ids = np.arange(macsis.num_zooms)

        bias_selection = (
            (m200 * 1E10 > 10 ** (14.5) / 0.6777) &
            (fgas > 0.05)
        )
        unbiased_mask = np.where(bias_selection)[0]  
        self.m500 = m500[unbiased_mask]
        self.macsis_ids = macsis_ids[unbiased_mask]
        
        self.redshift_value = {}
        self.mass_functions = {}
        for redshift_index in redshift_index_list:
            self.mass_functions[redshift_index] = self.get_mass_function(redshift_index)
            self.redshift_value[redshift_index] = macsis.get_zoom(0).get_redshift(redshift_index).redshift
        
        # self.plot_mass_functions()
        self.get_intersection_extrema()
        
        # Fit intersection histogram
        p0 = [np.max(self.intersection), np.median(self.bin_centres), np.std(self.bin_centres)]
        popt, _ = curve_fit(self.gaussian, self.bin_centres, self.intersection, p0=p0)        
        self.weighting_function = lambda log_mass: self.gaussian(log_mass, *popt)
        self.weight_normalisation = popt[0]
        
        self.find_matches(add_to_plot=True)
        
    
    @staticmethod
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    def get_mass_function(self, redshift_index: int) -> np.ndarray:
        filename = f"properties_0{redshift_index:d}.hdf5"
        m500_redshift = load_dict_from_hdf5(f"{macsis.output_dir:s}/{filename:s}")['m_500crit'] * 1E10
        m500_redshift = np.log10(m500_redshift[self.macsis_ids])    
        return np.histogram(m500_redshift, bins=self.bins)[0]
        
    def plot_mass_functions(self) -> None:        
        for redshift_index, mass_function in self.mass_functions.items():
            label = f"z = {self.redshift_value[redshift_index]:.1f}"
            plt.fill_between(self.bin_centres, mass_function, alpha=0.5, linewidth=0, label=label, step='mid', color=next(color_palette))
            
    def get_intersection_extrema(self) -> None:                
        self.intersection = np.minimum(
            self.mass_functions[self.min_max_redshifts[0]], 
            self.mass_functions[self.min_max_redshifts[1]]
        )
        self.intersection_domain_start = np.min(self.bins[:-1][self.intersection > 0])
        self.intersection_domain_end = np.max(self.bins[1:][self.intersection > 0])


    def find_matches(self, add_to_plot: bool = False) -> None:
        
        self.m500_sampled = {}
        self.ids_sampled = {}
        self.num_matches = {}
        self.mass_function_resampled = {}
        
        for redshift_index in self.redshift_index_list:    
            filename = f"properties_0{redshift_index:d}.hdf5"
            m500_redshift = load_dict_from_hdf5(f"{macsis.output_dir:s}/{filename:s}")['m_500crit'] * 1E10
            m500_redshift = np.log10(m500_redshift[self.macsis_ids])    

            # Create histogram with only masses in the domain of the intersection    
            m500_redshift_overlap_domain = np.where(
                (m500_redshift > self.intersection_domain_start) &
                (m500_redshift < self.intersection_domain_end)
            )[0]
            m500_redshift = m500_redshift[m500_redshift_overlap_domain]
            macsis_ids_redshift = self.macsis_ids[m500_redshift_overlap_domain]

            # Compute weighting function
            weights = self.weighting_function(m500_redshift)
            weights /= weights.max()
            
            if redshift_index > np.median(self.redshift_index_list):
                masses_range_peak = m500_redshift[np.argmax(weights)]
                weights[np.where(m500_redshift < masses_range_peak)] = 1.
            elif redshift_index < np.median(self.redshift_index_list):
                masses_range_peak = m500_redshift[np.argmax(weights)]
                weights[np.where(m500_redshift > masses_range_peak)] = 1.
            
            # Sample the mass function which matched the intersection domain
            resampled = np.histogram(m500_redshift, bins=self.bins, weights=weights ** 2)[0]
            # resampled /= np.max(resampled)
            # resampled *= self.weight_normalisation
            
            if redshift_index < np.median(self.redshift_index_list):
                masses_range_peak = self.bin_centres[np.argmax(self.intersection)]
                resampled[np.where(self.bin_centres < masses_range_peak)] /= np.max(resampled)
                resampled[np.where(self.bin_centres < masses_range_peak)] *= self.weight_normalisation
            elif redshift_index > np.median(self.redshift_index_list):
                masses_range_peak = self.bin_centres[np.argmax(self.intersection)]
                resampled[np.where(self.bin_centres >= masses_range_peak)] /= np.max(resampled)
                resampled[np.where(self.bin_centres >= masses_range_peak)] *= self.weight_normalisation            
            resampled = np.floor(resampled).astype(int)

            full_count = np.histogram(m500_redshift, bins=self.bins)[0]

            m500_sampled = []
            ids_sampled = []
            for bin_start, bin_end, num_to_sample, num_total in zip(self.bins[:-1], self.bins[1:], resampled, full_count):

                if num_to_sample == 0 or num_total == 0:
                    ids_sampled.append([])
                    continue
                
                bin_mask = np.where((m500_redshift > bin_start) & (m500_redshift < bin_end))[0]
                m500_in_bin = m500_redshift[bin_mask]
                bin_object_ids = macsis_ids_redshift[bin_mask]

                rnd_sample = np.random.choice(num_total, num_to_sample, replace=False)
                bin_object_ids = bin_object_ids[rnd_sample]
                mass_sampled = m500_in_bin[rnd_sample]

                m500_sampled += mass_sampled.tolist()
                ids_sampled.append(bin_object_ids.tolist())


            resampled = np.histogram(np.asarray(m500_sampled), bins=self.bins)[0]
            
            if add_to_plot:
                label = f"Sampled catalogue at z = {self.redshift_value[redshift_index]:.1f}"
                plt.step(self.bin_centres, resampled, lw=0.5, label=label, where='mid', color=next(color_palette))
                
            self.mass_function_resampled[redshift_index] = resampled
            self.m500_sampled[redshift_index] = m500_sampled
            self.ids_sampled[redshift_index] = ids_sampled
            self.num_matches[redshift_index] = len([item for sublist in ids_sampled for item in sublist])
    
# redshift_matcher = MassMatchedRedshiftSampler([13, 14, 15, 16, 17, 22], nbins=30)
redshift_matcher = MassMatchedRedshiftSampler([13, 22], nbins=30)
plt.clf()
print('Number of clusters sampled =====>', redshift_matcher.num_matches.values())
print('ID of clusters sampled =====>', redshift_matcher.ids_sampled)

from scipy.ndimage import gaussian_filter1d


fig, ax = plt.subplots()
color_palette = '#003f5c', '#bc5090', '#ffa600'


for redshift_idx, mass_function in redshift_matcher.mass_functions.items():
    mass_function_smooth = gaussian_filter1d(mass_function, 0.01)
    
    if redshift_idx not in [13, 22]:    
        ax.step(redshift_matcher.bin_centres, mass_function_smooth, where='mid', lw=0.5, alpha=0, color='grey', ls=':')
    elif redshift_idx == 13:
        ax.step(redshift_matcher.bin_centres, mass_function_smooth, where='mid', lw=1.1, color='#003f5c', label=r"HMF at $z=0$")
    elif redshift_idx == 22:
        ax.step(redshift_matcher.bin_centres, mass_function_smooth, where='mid', lw=1.1, color='#bc5090', label=r"HMF at $z=1$")

ax.fill_between(redshift_matcher.bin_centres, redshift_matcher.intersection, alpha=0.5, linewidth=0, step='mid', color='#ffa600', label='Intersection')

# Plot weighting functions
function_x = np.linspace(*ax.get_xlim(), 300)

function_y = redshift_matcher.weighting_function(function_x)
masses_range_peak = function_x[np.argmax(function_y)]
function_y[np.where(function_x > masses_range_peak)] = function_y.max()
ax.plot(function_x, function_y, ls=':', color='#003f5c', label=r"$w\,(M_{500})$ for $z=1$")

function_y = redshift_matcher.weighting_function(function_x)
masses_range_peak = function_x[np.argmax(function_y)]
function_y[np.where(function_x < masses_range_peak)] = function_y.max()
ax.plot(function_x, function_y, ls=':', color='#bc5090', label=r"$w\,(M_{500})$ for $z=0$")

    
ax.set_ylim(0, 65)
ax.legend(handlelength=2.5)
ax.set_xlabel(r'$\log_{10}\,(M_{500} / {\rm M}_{\odot})$')
ax.set_ylabel('Number of clusters')

axins = ax.inset_axes([1.2, 0., 1., 1.])
axins.set_xlim(redshift_matcher.intersection_domain_start - 0.2, redshift_matcher.intersection_domain_end + 0.2)
axins.set_ylim(0, 27)

axins.fill_between(redshift_matcher.bin_centres, redshift_matcher.intersection, alpha=0.5, linewidth=0, step='mid', color='#ffa600')

function_x = np.linspace(*axins.get_xlim(), 100)

function_y = redshift_matcher.weighting_function(function_x)
masses_range_peak = function_x[np.argmax(function_y)]
function_y[np.where(function_x > masses_range_peak)] = function_y.max()
axins.plot(function_x, function_y, ls=':', color='#003f5c')

function_y = redshift_matcher.weighting_function(function_x)
masses_range_peak = function_x[np.argmax(function_y)]
function_y[np.where(function_x < masses_range_peak)] = function_y.max()
axins.plot(function_x, function_y, ls=':', color='#bc5090')

for i in range(1, 5):
    axins.axhline(function_y.max() / 4 * i, lw=0.5, color='grey', ls='--', alpha=0.7)
    axins.annotate(
        text=f'$w=${1 / 4 * i * 100:.0f}%',
        xy=(axins.get_xlim()[-1] - 0.05, function_y.max() / 4 * i + 0.1),
        color="k",
        ha="right",
        va="bottom",
        alpha=0.7,
    )
axins.set_xlabel(r'$\log_{10}\,(M_{500} / {\rm M}_{\odot})$')
axins.set_ylabel('Number of clusters')

for redshift_idx, mass_function_resampled in redshift_matcher.mass_function_resampled.items():
    mass_function_smooth = gaussian_filter1d(mass_function_resampled, 0.01)
    
    if redshift_idx not in [13, 22]:    
        axins.step(redshift_matcher.bin_centres, mass_function_smooth, where='mid', lw=0.5, alpha=0, color='grey', ls=':')
    elif redshift_idx == 13:
        axins.step(redshift_matcher.bin_centres, mass_function_smooth, where='mid', lw=1.1, color='#003f5c', label=r"Subset at $z=0$: " + f"{redshift_matcher.num_matches[redshift_idx]:d} clusters")
    elif redshift_idx == 22:
        axins.step(redshift_matcher.bin_centres, mass_function_smooth, where='mid', lw=1.1, color='#bc5090', label=r"Subset at $z=1$: " + f"{redshift_matcher.num_matches[redshift_idx]:d} clusters")

        
axins.legend(ncol=2)
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.2, alpha=1)
plt.savefig(f"{macsis.output_dir}/redshift_hmf_match.pdf", bbox_extra_artists=(axins,), bbox_inches='tight')
