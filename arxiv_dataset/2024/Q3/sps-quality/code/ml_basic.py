# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:07:38 2024

@author: David J. Kedziora
"""

import autonoml as aml
import os

dir_data = "../data/ml"
dir_results = "../results/ml"

models = {"alpha": None,
          "ols": "autonoml.components.sklearn.LinearRegressor",
        #   "sgd": "autonoml.components.sklearn.SGDRegressor",
          # "plsr": "autonoml.components.sklearn.PLSRegressor",
          # "lwpr": "autonoml.components.custom_slmc.LocallyWeightedProjectionRegressor",
          # "rf": "autonoml.components.sklearn.RandomForestRegressor",
          # "xgb": "autonoml.components.sklearn.GradientBoostingRegressor",
          "omega": None}
          

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

    for id_model, class_model in models.items():

        if class_model is None:
            continue

        strategy = aml.import_strategy("./basic.strat")

        strategy.search_space[class_model]["Include"] = "y"
        strategy.do_hpo = True
        # if "Hpars" in strategy.search_space[class_model]:
        #     strategy.do_hpo = True
        # else:
        #     strategy.do_defaults = True

        proj = aml.AutonoMachine(do_mp = False)

        for experimental_context in experimental_contexts:
            if not experimental_context is None:
                proj.ingest_file(in_filepath = "%s/%s_%s.csv" % (dir_data, substring, experimental_context),
                                in_tags = {"context": experimental_context})
                proj.query_with_file(in_filepath = "%s/%s_%s.csv" % (dir_data, substring, experimental_context), 
                                    in_tags = {"context": experimental_context})
                        
        if os.path.exists("%s/basic_%s" % (dir_results, id_model)):
            # If there is a previous run, import old pipelines and compare.
            # Note: For multiple reruns, the new folder must be rewritten.
            proj.learn(in_key_target = "g_best",
                    in_keys_features = ["events", "rate_bg", "rate_env", "decay_peak", "delay_mpe", "g_fit"], do_exclude = True,
                    do_immediate_responses = False,
                    in_strategy = strategy,
                    in_tags_allocation = [("context", aml.AllocationMethod.LEAVE_ONE_OUT)],
                    in_directory_results = "%s/basic_%s_new" % (dir_results, id_model),
                    in_directory_import = "%s/basic_%s/pipelines" % (dir_results, id_model),
                    in_import_allocation = {"1p2uW_3000cps": ("context", "1p2uW_3000cps", aml.AllocationMethod.LEAVE_ONE_OUT),
                                            "2p5uW_4000cps": ("context", "2p5uW_4000cps", aml.AllocationMethod.LEAVE_ONE_OUT),
                                            "4uW_4100cps": ("context", "4uW_4100cps", aml.AllocationMethod.LEAVE_ONE_OUT),
                                            "8uW_5100cps": ("context", "8uW_5100cps", aml.AllocationMethod.LEAVE_ONE_OUT),
                                            "10uW_6000cps": ("context", "10uW_6000cps", aml.AllocationMethod.LEAVE_ONE_OUT),
                                            "10uW_12000cps": ("context", "10uW_12000cps", aml.AllocationMethod.LEAVE_ONE_OUT),
                                            "20uW_7000cps": ("context", "20uW_7000cps", aml.AllocationMethod.LEAVE_ONE_OUT),
                                            "30uW_7000cps": ("context", "30uW_7000cps", aml.AllocationMethod.LEAVE_ONE_OUT)})
        else:
            proj.learn(in_key_target = "g_best",
                    in_keys_features = ["events", "rate_bg", "rate_env", "decay_peak", "delay_mpe", "g_fit"], do_exclude = True,
                    do_immediate_responses = False,
                    in_strategy = strategy,
                    in_tags_allocation = [("context", aml.AllocationMethod.LEAVE_ONE_OUT)],
                    in_directory_results = "%s/basic_%s" % (dir_results, id_model))