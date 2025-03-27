from __future__ import print_function
from builtins import range
import numpy as np
from cosmosis.datablock import option_section, names as section_names


def setup(options):
    # only one parameter - filepath
    filename = options[option_section, "filepath"]
    des_fmt = options.get_bool(option_section, "des_fmt", default=False)
    histogram = options.get_bool(option_section, "histogram", default=False)

    output_section = options.get_string(
        option_section, "output_section", default=section_names.wl_number_density)
    single_bin = options.get_int(option_section, "single_bin", default=-666)
    upsampling = options.get_int(option_section, "upsampling", default=1)

    data_full = np.loadtxt(filename).T

    if des_fmt:
        z = 0.5 * (data_full[0] + data_full[1])
        nnz = len(z)
        nnzDESI = len(data_full) - 3
        nn_of_z = data_full[2:-1]
    else:
        nnz = len(data_full[0])
        nnzDESI = len(data_full) - 1
        z = data_full[0]
        if single_bin != -666:
            nn_of_z = data_full[single_bin]
            nnzDESI = 1
        else:
            nn_of_z = data_full[1:]

    if histogram:
        # in this case the sample z values are lower edges of
        # histogram bins.  So to turn them into samples we need to
        # shift everything.  This assumes equal sized bins
        dz = (z[1] - z[0]) / 2.0
        print("n(z) set to histogram mode. Bin centers are %f higher than edges." % dz)
        z += dz

    # check first z is zero, if not add some
    if z[0] > 0.00000001:
        z_new = np.zeros(len(z) + 1)
        z_new[1:] = z
        nn_of_z_new = np.zeros((nnzDESI, len(z) + 1))
        nn_of_z_new[:, 1:] = nn_of_z
        z, nn_of_z = z_new, nn_of_z_new

    if upsampling > 1:
        print("Upsampling z by factor {}".format(upsampling))
        z_new = np.linspace(0.0, z[-1], len(z) * upsampling)
        sample_bin = np.digitize(z_new, z) - 1
        nn_of_z_new = np.zeros((nnzDESI, len(z_new)))
        for i in range(nnzDESI):
            nn_of_z_new[i][:] = nn_of_z[i][sample_bin]
        z, nn_of_z = z_new, nn_of_z_new

    # Normalize n(z)
    for col in nn_of_z:
        norm = np.trapz(col, z)
        col /= norm

    print("Found %d samples and %d bins in redshift in file %s" % (nnzDESI, nnz, filename))
    return (nnz, nnzDESI, z, nn_of_z, output_section)


def execute(block, config):
    (nnz, nnzDESI, z, nn_of_z, output_section) = config

    block[output_section, 'nnz'] = nnz
    block[output_section, 'nnzDESI'] = nnzDESI
    block[output_section, 'z'] = z

    for (bin, nDESI_nn_of_z) in enumerate(nn_of_z):
        name = "n_DESI_%d" % (bin + 1)
        block[output_section, name] = nDESI_nn_of_z

    return 0


def cleanup(config):
    # nothing to do here!  We just include this
    # for completeness
    return 0
