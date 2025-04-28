import os
import numpy as np
import h5py
import bilby
from tqdm import tqdm

from . import logger, _default_postprocessed_phase_dir, _default_postprocessed_phase_filename_str
from . import gwphase
from . import gw_utils as gwutils
from .pe_input import ParameterEstimationInput

#Post-processed phases
_variables_event_1 = [
    'phase_H',
    'phase_L',
    'phase_V',
    'Dphi_f',
    'tau_HL',
    'tau_HV',
    'tau_LV'
]
#For event 2 we need to compute the phases in the frame of event 1
_variables_event_2 = [
    'phase_fbest',
    'a22_fbest',
    'zeta_fbest',
    'r_fbest',
    'phase_fhigh',
    'phase_flow',
    'tau_H',
    'tau_L',
    'tau_V',
    'ra',
    'dec',
    'psi',
    'geocent_time'
]

class PostprocessedPhase:
    def __init__(
        self,
        dataset,
        flow,
        fhigh,
        fbest,
        superevent_name=None,
        label=None,
    ):
        """
        A class to store the postprocessed phase

        Parameters
        ----------
        dataset: dict
            A dictionary containing the postprocessed phase data
        flow: float
            Lower frequency cutoff for computing :math:`\Delta \phi_f`
        fhigh: float
            Upper frequency cutoff for computing :math:`\Delta \phi_f`
        fbest: float
            Frequency at which the phase is best measured
        superevent_name: str
            Name of the superevent
        label: str
            Label for the postprocessed phase

        Returns
        -------
        PostprocessedPhase
            An instance of PostprocessedPhase class

        """
        # Make a list of variables that should exist in dataset
        variables = set([*_variables_event_1, *_variables_event_2])
        missing_keys = [p for p in dataset.keys() if p not in variables]
        assert missing_keys == [], f"Key(s) {missing_keys} is/are missing"

        self.dataset = dataset
        self.flow = flow
        self.fhigh = fhigh
        self.fbest = fbest
        self.superevent_name = superevent_name
        self.label = label

    @classmethod
    def from_file(cls, hdf5_file):
        """
        Load the postprocessed phase from a hdf5 file

        Parameters
        ----------
        hdf5_file: str
            Path to the hdf5 file

        Returns
        -------
        PostprocessedPhase
            An instance of PostprocessedPhase class
        
        """
        with h5py.File(hdf5_file, "r") as f:
            dataset = {p: np.array(f[p]) for p in [*_variables_event_1, *_variables_event_2]}

            flow = f.attrs['flow']
            fhigh = f.attrs['fhigh']
            fbest = f.attrs['fbest']
            superevent_name = None
            label = None
            if "superevent_name" in f.attrs.keys():
                superevent_name = f.attrs['superevent_name']
            if "label" in f.attrs.keys():
                label = f.attrs['label']

        return cls(
            dataset,
            flow,
            fhigh,
            fbest,
            superevent_name=superevent_name,
            label=label
        )

def postprocess_phase(
        pe_result,
        flow=20.,
        fhigh=100.,
        fbest=40.,
        superevent_name=None,
        label=None,
        output_dir=_default_postprocessed_phase_dir,
        output_filename=None,
    ):
    """
    Postprocess the phase of the GW signal.

    The procedure is described in Sec. II of https://arxiv.org/pdf/2308.06616.pdf 

    Parameters
    ----------
    pe_result: str or ParameterEstimationInput
        Path to the bilby result/PESummary file or an instance of ParameterEstimationInput
    flow: float
        Lower frequency cutoff for computing :math:`\Delta \phi_f`
    fhigh: float
        Upper frequency cutoff for computing :math:`\Delta \phi_f`
    fbest: float
        Frequency at which the phase is best measured
    superevent_name: str
        Name of the superevent
    label: str
        Label for the postprocessed phase
    output_dir: str
        Path to the output directory
    output_filename: str
        Name of the output file

    Returns
    -------
    PostprocessedPhase
        An instance of PostprocessedPhase class

    """

    if type(pe_result) is ParameterEstimationInput:
        ip = pe_result # No need to do anything
    elif type(pe_result) is str:
        try:
            # Maybe it is a bilby result file?
            ip = ParameterEstimationInput.from_bilby_result_file(pe_result)
        except:
            try:
                # Maybe it is a PESummary file?
                ip = ParameterEstimationInput.from_PESummary_file(pe_result)
            except:
                raise ValueError(f"Does not understand {pe_result}")

    posterior_samples = ip.posterior_samples
    fref = ip.reference_frequency
    duration = ip.duration 
    sampling_frequency = ip.sampling_frequency
    ifos = ip.ifo_list
    logger.info(f"Detectors online {ifos}")
    approx = ip.waveform_approximant
    logger.info(f"Waveform approximant {approx}")

    N_posteriors = len(posterior_samples['mass_1'])

    modes = [[2,2],[2,-2]]

    waveform_arguments = dict(
        waveform_approximant=approx,
        reference_frequency=fref,
        minimum_frequency=min(fref,flow) - 2./duration,
        maximum_frequency=fhigh + 2./duration,
        mode_array = modes
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    ra = posterior_samples['ra']
    dec = posterior_samples['dec']
    psi = posterior_samples['psi']
    geocent_time = posterior_samples['geocent_time']
    phase_fbest = np.zeros(N_posteriors)
    a22_fbest = np.zeros(N_posteriors)
    zeta_fbest = np.zeros(N_posteriors)
    r_fbest = np.zeros(N_posteriors)
    phase_fhigh = np.zeros(N_posteriors)
    phase_flow = np.zeros(N_posteriors)

    fs = bilby.core.utils.series.create_frequency_series(sampling_frequency,duration)
    rang_fs = (fs >= flow) & (fs <= fhigh)
    i_best = np.argmin(abs(fs[rang_fs]-fbest))

    for i in tqdm(range(N_posteriors)):
        phase_fbest[i], a22_fbest[i], zeta_fbest[i], r_fbest[i], phase_fhigh[i], phase_flow[i] = gwphase.phases_fs(
            waveform_generator,
            detec = "H1",
            rang_fs = rang_fs,
            i_best = i_best,
            mass_1=posterior_samples['mass_1'][i],
            mass_2=posterior_samples['mass_2'][i],
            a_1=posterior_samples['a_1'][i],
            a_2=posterior_samples['a_2'][i],
            tilt_1=posterior_samples['tilt_1'][i],
            tilt_2=posterior_samples['tilt_2'][i],
            phi_12=posterior_samples['phi_12'][i],
            phi_jl=posterior_samples['phi_jl'][i],
            luminosity_distance=posterior_samples['luminosity_distance'][i],
            theta_jn=posterior_samples['theta_jn'][i],
            psi=psi[i],
            phase_ref=posterior_samples['phase'][i],
            geocent_time=geocent_time[i],
            ra=ra[i],
            dec=dec[i],
            duration = duration,
            sampling_frequency = sampling_frequency
        )

    time_delay_H = np.zeros(N_posteriors)
    time_delay_L = np.zeros(N_posteriors)
    time_delay_V = np.zeros(N_posteriors)
    for i in range(N_posteriors):
        time_delay_H[i] = gwutils.time_delay_det("H1", ra[i], dec[i], geocent_time[i])
        time_delay_L[i] = gwutils.time_delay_det("L1", ra[i], dec[i], geocent_time[i])
        time_delay_V[i] = gwutils.time_delay_det("V1", ra[i], dec[i], geocent_time[i])

    #Arrival time phase at each detector
    tau_H = 2.*np.pi*fbest*(geocent_time + time_delay_H)
    tau_L = 2.*np.pi*fbest*(geocent_time + time_delay_L)
    tau_V = 2.*np.pi*fbest*(geocent_time + time_delay_V)

    #Arrival time phase difference between detectors. 
    #See Eq. 4 of https://arxiv.org/pdf/2308.06616.pdf
    tau_HL = tau_H-tau_L
    tau_HV = tau_H-tau_V
    tau_LV = tau_L-tau_V

    #Antenna patterns at each detector
    Fp_H, Fx_H = gwutils.FpFx("H1", ra, dec, psi, geocent_time)
    Fp_L, Fx_L = gwutils.FpFx("L1", ra, dec, psi, geocent_time)
    Fp_V, Fx_V = gwutils.FpFx("V1", ra, dec, psi, geocent_time)

    #Detector phases. 
    #See Eq. 2 of https://arxiv.org/pdf/2308.06616.pdf
    phase_H = gwphase.phase_d(phase_fbest,a22_fbest,zeta_fbest,Fp_H,Fx_H)
    phase_L = gwphase.phase_d(phase_fbest,a22_fbest,zeta_fbest,Fp_L,Fx_L)
    phase_V = gwphase.phase_d(phase_fbest,a22_fbest,zeta_fbest,Fp_V,Fx_V)

    #Detector phase evolution in frequency
    #See paragraph below Eq. 2 of https://arxiv.org/pdf/2308.06616.pdf
    Dphi_f = phase_fhigh - phase_flow

    # Saving the data
    variables = [*_variables_event_1, *_variables_event_2]
    if label is None:
        if superevent_name is not None:
                label_for_filename = superevent_name
        else:
            if type(pe_result) is str:
                # Try to figure out the proper label from filename
                label_for_filename = os.path.basename(pe_result).split('.')[0].split('_')[0]
            else:
                # Can't do much at this point......
                label_for_filename = "event"
    else:
        if superevent_name is not None:
            label_for_filename = f"{superevent_name}_{label}"

    logger.info(f"Assigning {label} as the label")
    if output_filename is None:
        output_filename = _default_postprocessed_phase_filename_str.format(
            label_for_filename,
            fbest,
            fhigh,
            flow
        )

    output_fullpath = os.path.join(output_dir, output_filename)
    with h5py.File(output_fullpath, "w") as f:
        for var in variables:
            _ = f.create_dataset(var, data=eval(var))
        
        f.attrs["fbest"] = fbest
        f.attrs["fhigh"] = fhigh
        f.attrs["flow"] = flow
        if label is not None:
            f.attrs["label"] = label
        if superevent_name is not None:
            f.attrs["superevent_name"] = superevent_name
    
    logger.info(f"Postprocessing completed and saved to {output_fullpath}")

    return PostprocessedPhase.from_file(output_fullpath)