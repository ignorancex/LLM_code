# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:02:44 2024

@author: David J. Kedziora
"""

import autonoml as aml

import csv
import time
import numpy as np

# The value to multiply learning rate by.
# multiplier = 50
# multiplier = 10000
# multiplier = 0.3

# Reset the model.
do_reset = False
# do_reset = True

dir_data = "../data/ml"
# model_type = "basic"
model_type = "synth"
# multiplier = 1
multiplier = 0.5

dir_models = "../results/ml/" + model_type + "_sgd"
dir_results = "../results/ml/adapt_" + model_type + "_sgd_" + f"{multiplier:.0e}"

print(dir_results)

substring = "sps_cumsum_norm"

experimental_contexts = [None,
                        "1p2uW_3000cps",
                        "2p5uW_4000cps",
                        "4uW_4100cps",
                        "8uW_5100cps",
                        "10uW_6000cps",
                        "10uW_12000cps",
                        "20uW_7000cps",
                        "30uW_7000cps",
                        None]

if __name__ == '__main__':

    # Because there is no HPO or any new pipelines, the strategy file does not matter.
    strategy = aml.import_strategy("./adapt.strat")

    proj = aml.AutonoMachine(do_mp = False)

    extra_string = ""
    target = "g_best"
    if model_type == "synth":
        extra_string = "1000)&(context!="
        target = "best"

    proj.learn(in_key_target = target,
            in_keys_features = ["events", "rate_bg", "rate_env", "decay_peak", "delay_mpe", "g_fit"], do_exclude = True,
            in_strategy = strategy,
            in_directory_results = dir_results,
            in_directory_import = dir_models + "/pipelines",
            in_import_allocation = {extra_string + "1p2uW_3000cps": ("context", "1p2uW_3000cps"),
                                    extra_string + "2p5uW_4000cps": ("context", "2p5uW_4000cps"),
                                    extra_string + "4uW_4100cps": ("context", "4uW_4100cps"),
                                    extra_string + "8uW_5100cps": ("context", "8uW_5100cps"),
                                    extra_string + "10uW_6000cps": ("context", "10uW_6000cps"),
                                    extra_string + "10uW_12000cps": ("context", "10uW_12000cps"),
                                    extra_string + "20uW_7000cps": ("context", "20uW_7000cps"),
                                    extra_string + "30uW_7000cps": ("context", "30uW_7000cps")},
            do_only_allocation = True,
            do_compare_adaptation = True,
            do_adapt_to_everything = True,
            do_rerank_learners = False)

    # streamers = dict()

    # idx_context = 0
    field_names = None
    for experimental_context in experimental_contexts:
        if not experimental_context is None:

            if field_names is None:
                with open("%s/%s_%s.csv" % (dir_data, substring, experimental_context), "r") as file:
                    csv_reader = csv.reader(file)
                    field_names = next(csv_reader)

            # The models are trained with best as a target.
            # Instead, fake the fitted estimate as the best target...
            field_names[-1] = "g_best_true"
            if model_type == "synth":
                field_names[-2] = target
            else:
                field_names[-2] = target

            do_multiplier = True
            while do_multiplier:
                try:
                    if not len(proj.solver.solution.groups["(context==%s)" % experimental_context]) == 2:
                        raise Exception()
                    for pipeline in proj.solver.solution.groups["(context==%s)" % experimental_context]:
                        pipeline.components[-1].model.eta0 *= multiplier
                        print("%s learning rate multiplied by %i." % (pipeline.name, multiplier))
                        if do_reset:
                            # The standard scaler should restart the moment it sees any samples.
                            pipeline.components[0].model.scale_ *= 0
                            pipeline.components[0].model.scale_ += 1.0
                            pipeline.components[0].model.mean_ *= 0
                            pipeline.components[0].model.var_ *= 0
                            pipeline.components[0].model.var_ += 1.0
                            pipeline.components[0].model.n_samples_seen_ *= 0

                            pipeline.components[-1].model.coef_ *= 0
                            pipeline.components[-1].model.intercept_ *= 0
                            pipeline.components[-1].model.t_ = 1.0
                            print("%s multiplied by zero." % (pipeline.name))
                    do_multiplier = False
                except:
                    time.sleep(1)
                    pass

            proj.ingest_file(in_filepath = "%s/%s_%s.csv" % (dir_data, substring, experimental_context),
                            in_field_names = field_names,
                            in_tags = {"context": experimental_context})