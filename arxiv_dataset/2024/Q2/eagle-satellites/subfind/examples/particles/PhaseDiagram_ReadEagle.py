import numpy as np
import matplotlib.pyplot as plt 
from read_eagle import EagleSnapshot
from read_header import read_header
import h5py

class PhaseDiagram_ReadEagle:

    def __init__(self, gn, sgn, centre, load_region_length=2):

        # Load information from the header.
        self.a, self.h, self.boxsize = read_header()

        # Load data.
        self.gas = self.read_galaxy(0, gn, sgn, centre, load_region_length)

        # Plot.
        self.plot()

    def read_galaxy(self, itype, gn, sgn, centre, load_region_length):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the Temperature, Density and StarFormationRate of all gas particles
        using the read_eagle routine. Conversion factors are still loaded directly
        from the hdf5 files. """

        data = {}

        # Initialize read_eagle module.
        eagle_data = EagleSnapshot('./data/snap_028_z000p000.0.hdf5')

        # Put centre into cMpc/h units.
        centre *= self.h

        # Select region to load, a 'load_region_length' cMpc/h cube centred on 'centre'.
        region = np.array([
            (centre[0]-0.5*load_region_length), (centre[0]+0.5*load_region_length),
            (centre[1]-0.5*load_region_length), (centre[1]+0.5*load_region_length),
            (centre[2]-0.5*load_region_length), (centre[2]+0.5*load_region_length)
        ]) 
        eagle_data.select_region(*region)

        # Load data using read_eagle, load conversion factors manually.
        f = h5py.File('./data/snap_028_z000p000.0.hdf5', 'r')
        for att in ['GroupNumber', 'SubGroupNumber', 'Temperature', 'Density', 'StarFormationRate']:
            tmp  = eagle_data.read_dataset(itype, att)
            cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
            aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
            hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
            data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
        f.close()

        # Mask to selected GroupNumber and SubGroupNumber.
        mask = np.logical_and(data['GroupNumber'] == gn, data['SubGroupNumber'] == sgn)
        for att in data.keys():
            data[att] = data[att][mask]

        return data

    def plot(self):
        """ Plot Temperature--Density relation. """
        plt.figure()

        # Plot currently star forming gas red.
        mask = np.where(self.gas['StarFormationRate'] > 0)
        plt.scatter(np.log10(self.gas['Density'][mask]), np.log10(self.gas['Temperature'][mask]),
            c='red', s=3, edgecolor='none')

        # Plot currently non star forming gas blue.
        mask = np.where(self.gas['StarFormationRate'] == 0)
        plt.scatter(np.log10(self.gas['Density'][mask]), np.log10(self.gas['Temperature'][mask]),
            c='blue', s=3, edgecolor='none')

        # Save plot.
        plt.minorticks_on()
        plt.ylabel('log10 Temperature [K]'); plt.xlabel('log10 Density [g/cm**3]')
        plt.tight_layout()
        plt.savefig('PhaseDiagram_ReadEagle.png')
        plt.close()

if __name__ == '__main__':
    # Centre is the COP for GN=1 SGN=0 taken from the database.
    centre = np.array([12.08808994,4.47437191,1.41333473])  # cMpc
    x = PhaseDiagram_ReadEagle(1, 0, centre)

