import numpy as np
import matplotlib.pyplot as plt
sigma = 0.15 # 0.15 kpc, for the height distribution of the disk
FOLDER_OUTPUT = 'data/plots/models_galaxy'


def z(r, b): # r is the distance from the Galactic center, b is the Galactic latitude
    return r*np.sin(b)


def height_distribution(z): # z is the height above the Galactic plane
    """
    Args:
        z: height above the Galactic plane in kpc
    Returns:
        height distribution of the disk at a height z above the Galactic plane
    """
    return np.exp(-0.5 * z**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def plot_density_heigth_distribution():
    """ Plot the density distribution of the disk as a function of height above the plane

    Returns:
        None. Saves the plot
    """
    zs = np.arange(0, 2, 0.01)
    density = height_distribution(zs)
    selected_zs = np.linspace(0.2, 0.7, 6, endpoint=True)
    selected_densities = height_distribution(selected_zs)
    for i, z in enumerate(selected_zs):
        val = selected_densities[i]
        plt.scatter(z, val, s=30, label=f"$z$ = {z} kpc, $P_z(z)$ = {val:.2e}", zorder=1)
    plt.plot(zs, density)
    plt.legend()
    #plt.title("Density distribution as a function of height above the plane")
    plt.xlabel("Distane $z$ (kpc) above the plane")
    plt.ylabel("$P_z(z)$")
    plt.xlim(0, 2)
    plt.ylim(bottom=0)
    plt.savefig(f'{FOLDER_OUTPUT}/density_distribution_z.pdf')
    plt.close()


def plot_line_of_sight_lattitude():
    """ Plot the line of sight as a function of r for different values of lattitude b
    
    Returns:
        None. Saves the plot
    """
    r = np.arange(0, 40, 0.01)
    z5 = z(r, b=np.radians(5))
    z4 = z(r, b=np.radians(4))
    z3 = z(r, b=np.radians(3))
    z2 = z(r, b=np.radians(2))
    z1 = z(r, b=np.radians(1))
    plt.plot(r, z1, label="$b$ = 1 deg")
    plt.plot(r, z2, label="$b$ = 2 deg")
    plt.plot(r, z3, label="$b$ = 3 deg")
    plt.plot(r, z4, label="$b$ = 4 deg")
    plt.plot(r, z5, label="$b$ = 5 deg")
    plt.plot(r, 0.5*np.ones(len(r)), c='black', label="$0-$density", linestyle='--')
    plt.legend()
    #plt.title("Line of sight as a function of r for different values of lattitude b")
    plt.xlabel("Distance $r$ (kpc)")
    plt.ylabel("Distance $z$ (kpc)")
    plt.xlim(0, 40)
    plt.ylim(bottom=0)
    plt.savefig(f'{FOLDER_OUTPUT}/line_of_sight_lattitude.pdf')
    plt.close()


def main():
    plot_density_heigth_distribution()
    plot_line_of_sight_lattitude()

if __name__ == "__main__":
    main()