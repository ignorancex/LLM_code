import h5py
import numpy as np
import matplotlib.pyplot as plt 
from read_dataset import read_dataset

class PhaseDiagram:

    def __init__(self, gn, sgn):

        # Load data.
        self.gas = self.read_galaxy(0, gn, sgn)

        # Plot.
        self.plot()

    def read_galaxy(self, itype, gn, sgn):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the temperature, density and star formation rate of all gas particles. """

        data = {}

        # Load data.
        for att in ['GroupNumber', 'SubGroupNumber', 'Temperature', 'Density', 'StarFormationRate']:
            data[att] = read_dataset(itype, att)
        
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
        plt.savefig('PhaseDiagram.png')
        plt.close()

if __name__ == '__main__':
    x = PhaseDiagram(1, 0)

