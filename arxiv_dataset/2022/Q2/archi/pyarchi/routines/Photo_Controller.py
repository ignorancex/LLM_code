import yaml
from functools import wraps

from .photometric_process import photometry
from pyarchi.data_objects import Data

from pyarchi.utils import general_optimizer, store_optimized_radius
from pyarchi.utils import create_logger, parameters_validator, handle_folders

logger = create_logger("Photo Controller")


class Photo_controller:
    """
        Controller class that allows an easy interface with all the important functions in the module. If the no_optim parameter is set to False,
        then it automatically starts the optimization routine, at instantiation time of the class.
    """

    def __init__(self, job_number, config_path="configuration_files/config.yaml", no_optim=False):
        self.job_number = job_number
        logger.info("Loading Parameters:")

        if isinstance(config_path, str):
            with open(config_path, "r") as stream:
                kwargs = yaml.load(stream)
        else:  # we can pass a dict
            kwargs = config_path

        self.kwargs = kwargs
        self.data_fits = Data(kwargs['base_folder'])

        if no_optim:
            self.kwargs["optimize"] = 0

        if self.kwargs[
            "optimize"
        ]:  # Finds the best factors to minimize Cv and updates the json file
            self.__optimize()

        self.master_save_folder = handle_folders(len(self.data_fits.stars), self.job_number, **kwargs)

        self._completed_run = False

    def _check_parameters(func):  # pylint: disable=no-self-argument
        """
        Used to validate the error flag before running a function.
        Decorator used inside this class

        Returns
        -------

        """

        @wraps(func)
        def on_call(self, *args, **kwargs):

            if (
                    func.__name__ == "run" and self.kwargs["optimize"]
            ):  # pylint: disable=no-member
                pass  # during optimization process there are no checks
            else:
                wrong_params, warnings, kwargs = parameters_validator(**self.kwargs)
                if any(warnings):
                    logger.fatal("\t Found Warnings: {}".format(len(warnings)))
                    [logger.fatal("\t \t" + warn) for warn in warnings]

                if any(wrong_params):
                    logger.fatal(
                        "Problem with the passed parameters. Process is going to stop now"
                    )
                    logger.fatal("\t Bad parameters: {}".format(len(wrong_params)))
                    [logger.fatal("\t \t" + wrong_param) for wrong_param in wrong_params]
                    return -1

            if func.__name__ == "__optimize":  # pylint: disable=no-member
                return func(self)  # pylint: disable=not-callable
            else:
                return func(self, *args, **kwargs)  # pylint: disable=not-callable

        return on_call

    @_check_parameters
    def run(self, DataFits=None, factor=None, **kwargs):
        """
        Calls the main method to run the star analysis pipeline

        Parameters
        ----------
        factor
            Used for the optimization process. During normal functioning process then so value should be passed
        kwargs
            Config values, retrieved from the file on the class initialization
        Returns
        -------
            data_fits: instance of :class:`~pyarchi.data_objects.Data.Data`, with the relevant information.
        """

        configs = (
            self.kwargs if (factor is None and not self.kwargs["optimize"]) else kwargs
        )

        data_fits = DataFits if DataFits is not None else self.data_fits
        result = data_fits.load_parameters(factor, **configs)

        if result == -1:
            logger.fatal("Critical error")
            return data_fits

        self.data_fits = photometry(data_fits=data_fits, save_folder=self.master_save_folder, **configs)

        if not self.kwargs["optimize"]:
            logger.info("Checking for out of bounds masks:")
            found = False
            for star in self.data_fits.stars:
                if star.out_bound:
                    logger.warning("\t\t Star {} is out of bounds".format(star.number))
                    found = True
            print("\t{}\n==============//==============".format("-> Found nothing" if not found else ''))
        if data_fits.abort_process:
            logger.fatal("Problems were found. Could not run properly !!!!!")
            raise Exception("Something went wrong")

        self._completed_run = True
        logger.info("Finished the run; Everything done !!")
        return self.data_fits

    def change_parameters(self, params_dict):
        """
        Updates the kwargs with new values. A dictionary is passed in, with keys corresponding to parameters and each
        value it's the new configuration for that specific parameter.
        At this stage no validation is made over the passed values. Such validation occurs before the routines, to make sure that the correct parameters were added.

        Parameters
        ----------
        params_dict:
            Dictionary with the values that we wish to change

        Returns
        -------

        """
        new_dict = self.kwargs.copy()
        for key, value in params_dict.items():
            if key not in self.kwargs:
                raise AttributeError(
                    "The {} configuration value does not exist".format(key)
                )
            if key == 'optimize' and value != 0:
                raise KeyError(
                    "Cannot set optimize parameter to one. Use the .optimize() method to manually trigger the optimization "
                    "routine "
                    )
            new_dict[key] = value

        self.kwargs = new_dict
        self.data_fits.used_file = self.kwargs["base_folder"]

    def optimize(self):
        """
        Run the optimization routine without instantiating a new object.
        Returns
        -------

        """
        self.kwargs["optimize"] = 1  # just making sure
        self.__optimize()

    @_check_parameters
    def __optimize(self):
        """
        Optimization process. Called when the class is initialized
        Returns
        -------

        """
        import time

        t0 = time.time()
        logger.info("Starting optimization process")

        vals = general_optimizer(
            self.run,
            data_f=Data('---'),
            job_number=self.job_number,
            max_process=self.kwargs["optim_processes"],
            **self.kwargs
        )
        if vals == -1:
            raise Exception("Something went wrong")
        logger.info("Optimization process completed after {} s".format(time.time() - t0))
        self.kwargs["optimize"] = 0

        if store_optimized_radius(vals, **self.kwargs) == -1:
            logger.critical("Problems were found. Failed to save optimized values.")
            return
        del vals
        self._completed_run = False

    @property
    def parameters(self):
        return self.kwargs
