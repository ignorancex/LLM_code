import h5py
import numpy as np
import struct
import os
from astropy.cosmology import Planck18 as cosmo
import scipy.interpolate
import re


z_table = np.linspace(0, 15, 3000)
d_L_table = cosmo.luminosity_distance(z_table)
zfunc = scipy.interpolate.interp1d(d_L_table, z_table)


def write_record(fp, typ, content):
    if type(content) == np.ndarray:
        content = list(content.flatten('F'))

    body = b''.join(
        struct.pack('<'+typ, i) for i in content
    )
    fp.write(struct.pack('<I', len(body)))
    fp.write(body)
    fp.write(struct.pack('<I', len(body)))


def convert_injection(ifar_find=1, fi='../endo3_mixture-LIGO-T2100113-v12.hdf5', fo='inj.rec', version=3):
    with h5py.File(fi, 'r') as f:
        if version == 2:
            mask = (
                np.array(f['injections/ifar_gstlal']) > ifar_find
            ) & (
                np.array(f['injections/ifar_pycbc_full']) > ifar_find
            ) & (
                np.array(f['injections/ifar_pycbc_bbh']) > ifar_find
            )

            s1  = f['injections/spin1z'][mask]
            s2  = f['injections/spin2z'][mask]

            m1D = f['injections/mass1_source'][mask] * (1+f['injections/redshift'][mask])
            m2D = f['injections/mass2_source'][mask] * (1+f['injections/redshift'][mask])
        elif version == 3:
            mask = [
                np.array(f['injections/ifar_gstlal']) > ifar_find,
                np.array(f['injections/ifar_pycbc_bbh']) > ifar_find
            ]
            if 'injections/ifar_pycbc_full' in f:
                mask.append(
                    np.array(f['injections/ifar_pycbc_full']) > ifar_find
                )
            mask = np.all(mask, axis=0)

            s1 = np.sqrt(f['injections/spin1x'][mask]**2 + f['injections/spin1y'][mask]**2 + f['injections/spin1z'][mask]**2)
            s2 = np.sqrt(f['injections/spin2x'][mask]**2 + f['injections/spin2y'][mask]**2 + f['injections/spin2z'][mask]**2)

            m1D = f['injections/mass1'][mask]
            m2D = f['injections/mass2'][mask]
        m1 = f['injections/mass1_source'][mask]
        m2 = f['injections/mass2_source'][mask]
        rs = f['injections/redshift'][mask]
        ld = f['injections/distance'][mask]
        s1z = f['injections/spin1z'][mask]
        s2z = f['injections/spin2z'][mask]

        dat = np.column_stack((
            m1, m2, m1D, m2D,
            rs, ld,
            s1, s2
        ))

    with open(fo, 'wb') as fp:
        write_record(fp, 'i', [len(dat)])
        write_record(fp, 'd', dat)


def load_gw(veto, base='../tmp/', v=2):
    if v == 1:
        pat = re.compile(r'GW([\d_]*)_GWTC-1.hdf5')
    elif v == 2:
        pat = re.compile(r'GW([\d_]*)_comoving.h5')
    elif v == 21:
        pat = re.compile(r'IGWN-GWTC2p1-v\d*-GW([\d_]*)_PEDataRelease.h5')
    elif v == 3:
        pat = re.compile(r'IGWN-GWTC3p0-v\d*-GW([\d_]*)_PEDataRelease_mixed_cosmo.h5')

    files = [pat.match(i) for i in os.listdir(base)]
    files = [
        base + i.group(0)
        for i in files
        if i and i.group(1) not in veto
    ]

    m1 = np.array([])
    m2 = np.array([])
    m1D = np.array([])
    m2D = np.array([])
    rs = np.array([])
    ld = np.array([])
    s1 = np.array([])
    s2 = np.array([])

    offsets = []

    for fn in files:
        print(v, fn)
        with h5py.File(fn, 'r') as f:
            if v == 1:
                d = f['Overall_posterior']

                rsi = zfunc(d['luminosity_distance_Mpc'])
                m1i = d['m1_detector_frame_Msun']
                m2i = d['m2_detector_frame_Msun']

                if np.mean(m2i) < 2.5 or np.mean(m1i) < 2.5:
                    print("Skipping", fn)
                    continue

                m1 = np.concatenate((m1, m1i / (1+rsi)))
                m2 = np.concatenate((m2, m2i / (1+rsi)))
                m1D = np.concatenate((m1D, m1i))
                m2D = np.concatenate((m2D, m2i))
                rs = np.concatenate((rs, rsi))
                ld = np.concatenate((ld, d['luminosity_distance_Mpc']))
                # ce = np.concatenate((
                #     ce,
                #     (
                #         d['spin1'] * m1i * d['costilt1']
                #         + d['spin2'] * m2i * d['costilt2']
                #     ) / (m1i + m2i)
                # ))
                s1 = np.concatenate((s1, d['spin1']))
                s2 = np.concatenate((s2, d['spin2']))

            elif v == 2:
                d = f['PublicationSamples/posterior_samples']
                # chi_eff = (spin1 * m1 * cos1 + spin2 * m2 * cos2)/(m1+m2)
                if np.mean(d['mass_1_source']) < 2.5 or np.mean(d['mass_2_source']) < 2.5:
                    print("Skipping", fn)
                    continue

                m1 = np.concatenate((m1, d['mass_1_source']))
                m2 = np.concatenate((m2, d['mass_2_source']))
                m1D = np.concatenate((m1D, d['mass_1']))
                m2D = np.concatenate((m2D, d['mass_2']))
                rs = np.concatenate((rs, d['redshift']))
                ld = np.concatenate((ld, d['luminosity_distance']))
                s1 = np.concatenate((s1, np.sqrt(d['spin_1x']**2 + d['spin_1y']**2 + d['spin_1z']**2)))
                s2 = np.concatenate((s2, np.sqrt(d['spin_2x']**2 + d['spin_2y']**2 + d['spin_2z']**2)))
            elif v == 21:
                d = f['PrecessingSpinIMRHM/posterior_samples']
                if np.mean(d['mass_1_source']) < 2.5 or np.mean(d['mass_2_source']) < 2.5:
                    print("Skipping", fn)
                    continue
                m1 = np.concatenate((m1, d['mass_1_source']))
                m2 = np.concatenate((m2, d['mass_2_source']))
                m1D = np.concatenate((m1D, d['mass_1']))
                m2D = np.concatenate((m2D, d['mass_2']))
                rs = np.concatenate((rs, d['redshift']))
                ld = np.concatenate((ld, d['luminosity_distance']))
                s1 = np.concatenate((s1, np.sqrt(d['spin_1x']**2 + d['spin_1y']**2 + d['spin_1z']**2)))
                s2 = np.concatenate((s2, np.sqrt(d['spin_2x']**2 + d['spin_2y']**2 + d['spin_2z']**2)))
            elif v == 3:
                d = f['C01:Mixed/posterior_samples']
                if np.mean(d['mass_1_source']) < 2.5 or np.mean(d['mass_2_source']) < 2.5:
                    print("Skipping", fn)
                    continue
                m1 = np.concatenate((m1, d['mass_1_source']))
                m2 = np.concatenate((m2, d['mass_2_source']))
                m1D = np.concatenate((m1D, d['mass_1']))
                m2D = np.concatenate((m2D, d['mass_2']))
                rs = np.concatenate((rs, d['redshift']))
                ld = np.concatenate((ld, d['luminosity_distance']))
                s1 = np.concatenate((s1, np.sqrt(d['spin_1x']**2 + d['spin_1y']**2 + d['spin_1z']**2)))
                s2 = np.concatenate((s2, np.sqrt(d['spin_2x']**2 + d['spin_2y']**2 + d['spin_2z']**2)))

            # print(fn, min(m2))
            offsets.append(len(m1))

    return np.column_stack((m1, m2, m1D, m2D, rs, ld, s1, s2)), np.array(offsets)


def get_veto():
    veto = set()

    # From 2104.02685
    veto.update([
        "170817",
        "190521",
        "190425",
        "190814",
        "190909_114149",
        "190719_215514",
        "190426_152155"
    ])

    # 2108.01045
    veto.update([
        "190425_081805",
        "190707_093326",
        "190720_000836",
        "190725_174728",
        "190728_064510",
        "190814_211039",
        "190924_021846",
        "190930_133541"
    ])

    # 2111.03634
    veto.update([
        "200105_162426",
        "200115_042309",
        "190426_152155"
    ])

    # 2111.03634
    veto.update([
        "170817",
        "190425",
        "200105",
        "200115",
        "190426",
        "190917"
    ])

    # others
    veto.update([
        "190917_114630", "191219_163120"
        #"200210_092254"
    ])

    return veto

def get_far_veto():
    # far veto
    veto = set([
        '200322_091133', # 140.0
        '200308_173609', # 2.4
        '200306_093714', # 24.0
        '200220_124850', # 30.0
        '200220_061928', # 6.8
        '200210_092254', # 1.2
        '200208_222617', # 4.8
        '191219_163120', # 4.0
        '191204_110529', # 3.3
        '191126_115259', # 3.2
        '191113_071753', # 26.0
        '190926_050336', # 1.1
        '190916_200658', # 4.7
        '190909_114149', # 26.1830194416
        '190514_065416', # 2.8
        '190426_190642', # 4.1
        '190424_180648', # 9.038785546924121
        '190403_051519', # 7.7
    ])
    veto.update([
            '170817', #weird, binary neutron star
        ])
    veto.update([
            '190814', #EMRI
            '190917', # remove explicitly
            '190426_152155'
        ])

    return veto


def convert_gw(fo='data.rec', base='../tmp/'):
    veto = get_far_veto()
    d1, o1 = load_gw(veto, base, v=1)

    d2, o2 = load_gw(veto, base, v=2)
    o2 += len(d1)

    d21, o21 = load_gw(veto, base, v=21)
    o21 += len(d1) + len(d2)

    d3, o3 = load_gw(veto, base, v=3)
    o3 += len(d1) + len(d2) + len(d21)

    d = np.concatenate((d1, d2, d21, d3))
    o = np.concatenate((o1, o2, o21, o3))

    print("\n".join(
        f"{name} in [{mi}, {ma}]"
        for name, mi, ma in zip(
            ["m1","m2","m1D","m2D","rs ", "ld ", "s1 ", "s2 "],
            np.min(d, axis=0),
            np.max(d, axis=0),
        )
    ))

    with open(fo, 'wb') as fp:
        write_record(fp, 'i', [len(d), len(o)])
        write_record(fp, 'i', o)
        write_record(fp, 'd', d)

    return d, o


def plot_population(do):
    d, o = do
    oo = np.concatenate(([0], o))

    hist([mean(d[i:j,0]) for i,j in zip(o[:-1], o[1:])], alpha=0.3)
    hist([mean(d[i:j,1]) for i,j in zip(o[:-1], o[1:])], alpha=0.3)
    xlabel("$M_i/M_\odot$")
    ylabel("$n$")
    legend([f"$M_{i}$" for i in [0,1]])


def histogram_all(base='../tmp/', cm=True):
    def avg(d, o):
        oo = np.concatenate(([0], o))
        return [np.mean(d[i:j,0]) for i,j in zip(oo[:-1], oo[1:])]

    veto = get_veto()
    d1, o1 = load_gw(veto, base, v=1)
    d2, o2 = load_gw(veto, base, v=2)
    d21, o21 = load_gw(veto, base, v=21)
    d3, o3 = load_gw(veto, base, v=3)

    avg1 = avg(d1,o1)
    avg2 = avg(d2,o2)
    avg21 = avg(d21,o21)
    avg3 = avg(d3,o3)

    # hist([avg1, avg2, avg21, avg3], stacked=True)
    # xlabel("$M_i/M_\odot$")
    # ylabel("$n$")
    # legend([f'GWTC-{i}' for i in [1, 2, '2.1', 3]])

    open('avg.txt', 'w').write('\n'.join([
        ','.join(str(i) for i in avg1),
        ','.join(str(i) for i in avg2),
        ','.join(str(i) for i in avg21),
        ','.join(str(i) for i in avg3)
    ]))

def convert_gw_201014533(fo='data.rec', base='../tmp/', cm=True):
    veto=set(['170817','190425','190814'])
    veto.update([
        '190426_152155',
        '190719_215514',
        '190909_114149'
    ])
    d1, o1 = load_gw(veto, base, v=1)
    d2, o2 = load_gw(veto, base, v=2)
    o2 += len(d1)

    d = np.concatenate((d1, d2))
    o = np.concatenate((o1, o2))

    with open(fo, 'wb') as fp:
        write_record(fp, 'i', [len(d), len(o)])
        write_record(fp, 'i', o)
        write_record(fp, 'd', d)


if __name__ == '__main__':
    convert_injection()
    convert_injection(fi='../o3a_bbhpop_inj_info.hdf', fo='inj-201014533.rec', version=2)
    convert_gw_201014533('data-201014533.rec')
    convert_gw()
