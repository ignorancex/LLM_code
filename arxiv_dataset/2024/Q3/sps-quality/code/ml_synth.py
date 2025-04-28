# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:52:35 2023

@author: David J. Kedziora
"""

import autonoml as aml
import os

dir_data = "../data/ml"
dir_results = "../results/ml"

models = {"alpha": None,
        #   "ols": "autonoml.components.sklearn.LinearRegressor",
          "sgd": "autonoml.components.sklearn.SGDRegressor",
        #   "plsr": "autonoml.components.sklearn.PLSRegressor",
        #   "rf": "autonoml.components.sklearn.RandomForestRegressor",
        #   "xgb": "autonoml.components.sklearn.GradientBoostingRegressor",
          "omega": None}

filename_substring = "sps_quality_1000_events"

list_num_events = [None,
                   "1000",
                   "10000",
                   "100000",
                   "1000000",
                   None]

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

        strategy = aml.import_strategy("./synth.strat")

        strategy.search_space[class_model]["Include"] = "y"
        strategy.do_hpo = True
        # strategy.do_defaults = True

        n_instances = 2000
        if id_model in ["rf", "xgb"]:
            n_instances = 400

        proj = aml.AutonoMachine(do_mp = False)

        import_allocation = dict()
        for num_events in list_num_events:
            if not num_events is None:
                for experimental_context in experimental_contexts:
                    if not experimental_context is None:
                        proj.ingest_file(in_filepath = "%s/train_sps_quality_%s_events_%s.csv" % (dir_data, num_events, experimental_context),
                                        in_tags = {"size_category": num_events, "context": experimental_context},
                                        in_n_instances = n_instances)
                        proj.query_with_file(in_filepath = "%s/test_sps_quality_%s_events_%s.csv" % (dir_data, num_events, experimental_context),
                                            in_tags = {"size_category": num_events, "context": experimental_context})
                        
                        substring = "(size_category==%s)&(context!=%s)" % (num_events, experimental_context)
                        import_allocation[substring] = [("size_category", num_events), ("context", experimental_context, aml.AllocationMethod.LEAVE_ONE_OUT)]
                
        # proj.info_storage()

        if os.path.exists("%s/synth_%s" % (dir_results, id_model)):

            # If there is a previous run, import old pipelines and compare.
            # Note: For multiple reruns, the new folder must be rewritten.
            proj.learn(in_key_target = "best",
                    in_keys_features = ["events", "estimate"], do_exclude = True,
                    do_immediate_responses = False,
                    in_strategy = strategy,
                    in_tags_allocation = ["size_category", ("context", aml.AllocationMethod.LEAVE_ONE_OUT)],
                    in_directory_results = "%s/synth_%s_new" % (dir_results, id_model),
                    in_directory_import = "%s/synth_%s/pipelines" % (dir_results, id_model),
                    in_import_allocation = import_allocation)
        else:
            proj.learn(in_key_target = "best",
                    in_keys_features = ["events", "estimate"], do_exclude = True,
                    do_immediate_responses = False,
                    in_strategy = strategy,
                    in_tags_allocation = ["size_category", ("context", aml.AllocationMethod.LEAVE_ONE_OUT)],
                    in_directory_results = "%s/synth_%s" % (dir_results, id_model))