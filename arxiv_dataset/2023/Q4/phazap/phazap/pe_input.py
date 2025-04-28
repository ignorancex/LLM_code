import numpy as np
import pandas as pd

_required_parameters = [
    "mass_1",
    "mass_2",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "luminosity_distance",
    "theta_jn",
    "phase", 
    "ra",
    "dec",
    "geocent_time",
    "psi",
]

class ParameterEstimationInput():
    def __init__(
        self,
        posterior_samples,
        reference_frequency,
        sampling_frequency,
        duration,
        waveform_approximant,
        ifo_list,
    ):
        """
        A class to store the input for parameter estimation

        Parameters
        ----------
        posterior_samples: dict
            A dictionary of posterior samples
        reference_frequency: float
            Reference frequency chosen for this set of samples
        sampling_frequency: float
            Sampling frequency
        duration: float
            Duration
        waveform_approximant: str
            Waveform approximant
        ifo_list: list
            List of interferometers

        Returns
        -------
        ParameterEstimationInput
            The ParameterEstimationInput instance

        """
        self.posterior_samples = posterior_samples
        self.reference_frequency = reference_frequency
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.waveform_approximant = waveform_approximant
        self.ifo_list = ifo_list
        
    @classmethod
    def from_bilby_result_file(cls, result_file):
        """
        Load the parameter estimation input from a bilby result file

        Parameters
        ----------
        result_file: str
            Path to the result file
        
        Returns
        -------
        ParameterEstimationInput
            The ParameterEstimationInput instance
        
        """
        import bilby
        import bilby_pipe
        r = bilby.result.read_in_result(result_file)
        
        # Extract run information
        fref = r.meta_data["likelihood"]["waveform_arguments"]["reference_frequency"]
        fs = r.meta_data["likelihood"]["sampling_frequency"]
        duration = r.meta_data["likelihood"]["duration"]
        waveform_approximant = r.meta_data["likelihood"]["waveform_arguments"]["waveform_approximant"]
        if r.meta_data["command_line_args"]["sampler"] == "parallel_bilby":
            ifos = r.meta_data["config_file"]["detectors"]
        else:
            ifos = bilby_pipe.utils.convert_detectors_input(r.meta_data["command_line_args"]["detectors"])

        # Extract only the parameters that we need
        posterior_samples = pd.DataFrame({p: r.posterior[p].to_numpy() for p in _required_parameters})

        return cls(
            posterior_samples,
            fref,
            fs,
            duration,
            waveform_approximant,
            ifos,
        )

    @classmethod
    def from_PESummary_file(cls, hdf5_file):
        """
        Load the parameter estimation input from a PESummary hdf5 file

        Parameters
        ----------
        hdf5_file: str
            Path to the PESummary file
        
        Returns
        -------
        ParameterEstimationInput
            The ParameterEstimationInput instance

        """
        # TODO: Re-write the whole thing to actually NOT use PESummary because it is just SLOW
        from pesummary.io import read
        r = read(hdf5_file, package="gw")

        # There could be multiple datasets present in the file
        _available_dataset = sorted([k for k in r.pe_algorithm.keys() if "Mixed" not in k])
        # Prefer IMRPhenom-family waveform over EOB-family waveform
        _tag = _available_dataset[0]

        if r.pe_algorithm[_tag] == "bilby" or "bilby" in str(r.config[_tag]):
            # Bilby
            fref = float(r.config[_tag]["config"]["reference-frequency"])
            waveform_approximant = r.config[_tag]["config"]["waveform-approximant"]
            fs = float(r.config[_tag]["config"]["sampling-frequency"])
            duration = float(r.config[_tag]["config"]["duration"])
            ifos = eval(r.config[_tag]["config"]["detectors"])
        elif r.pe_algorithm[_tag] == "lalinference" or "lalinference" in str(r.config[_tag]):
            # LALInference
            fref = float(r.config[_tag]["engine"]["fref"])
            fs = float(r.config[_tag]["engine"]["srate"])
            duration = float(r.config[_tag]["engine"]["seglen"])
            waveform_approximant = r.config[_tag]["engine"]["approx"]
            if "pseudo" in waveform_approximant:
                import lalsimulation
                # Genius
                waveform_approximant = \
                lalsimulation.GetStringFromApproximant(lalsimulation.GetApproximantFromString(waveform_approximant))
            ifos = eval(r.config[_tag]["analysis"]["ifos"])
        else:
            raise ValueError("Cannot parse the input file")

        # Extract only the parameters that we need
        # WARNING: PESummary is VERY slow for whatever reason
        # You know what, let us just load the file again with h5py
        import h5py
        f = h5py.File(hdf5_file, "r")
        posterior_samples = {p: np.array(f[_tag]["posterior_samples"][p]) for p in _required_parameters}
        f.close()

        return cls(
            posterior_samples,
            fref,
            fs,
            duration,
            waveform_approximant,
            ifos,
        )
