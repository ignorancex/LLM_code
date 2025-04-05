from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from dehb import DEHB


class DEHBOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
            {
                "param": {
                    "type": "int" | "bool" | "cat" | "ordinal" | "real",
                    "space": "log" | "logit" when "real"
                    "range": [low, high]
                    "values": for categorical
                }
            }
        """
        AbstractOptimizer.__init__(self, api_config)

        cs = ConfigurationSpace()
        for key, config in api_config.items():
            if config["type"] == "int":
                cs.add(UniformIntegerHyperparameter(key, *config["range"], log=config["space"] in ["log", "logit"]))
            if config["type"] == "float":
                cs.add(UniformFloatHyperparameter(key, config["range"][0], config["range"][1]))
            if config["type"] == "bool":
                cs.add(CategoricalHyperparameter(key, [False, True]))
            if config["type"] == "cat":
                cs.add(CategoricalHyperparameter(key, config["values"]))
            if config["type"] == "real":
                cs.add(
                    UniformFloatHyperparameter(
                        key,
                        config["range"][0],
                        config["range"][1],
                        log=config["space"].startswith("log")
                    )
                )

        self.dehb = DEHB(
            f=None,
            cs=cs,
            min_fidelity=1,
            max_fidelity=10,
            n_workers=1
        )

    def suggest(self, n_suggestions=1):
        """Get suggestion.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        return [self.dehb.ask()["config"]]

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        self.dehb.tell(X, y)


if __name__ == "__main__":
    experiment_main(DEHBOptimizer)
