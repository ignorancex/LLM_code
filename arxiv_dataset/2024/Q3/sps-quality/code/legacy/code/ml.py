# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:52:35 2023

@author: David J. Kedziora
"""

import autonoml as aml

# filepath_data = "./data/mixed_0101_abrupto.csv"
dir_data = "../data/ml"
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

    strategy = aml.import_strategy("./sps.strat")

    proj = aml.AutonoMachine(do_mp = False)

    for num_events in list_num_events:
        if not num_events is None:
            for experimental_context in experimental_contexts:
                if not experimental_context is None:
                    proj.ingest_file(in_filepath = "%s/train_sps_quality_%s_events_%s.csv" % (dir_data, num_events, experimental_context),
                                     in_tags = {"size_category": num_events, "context": experimental_context},
                                     in_n_instances = 400)
                    
    for num_events in list_num_events:
        if not num_events is None:
            for experimental_context in experimental_contexts:
                if not experimental_context is None:
                    proj.query_with_file(in_filepath = "%s/test_sps_quality_%s_events_%s.csv" % (dir_data, num_events, experimental_context),
                                         in_tags = {"size_category": num_events, "context": experimental_context})
            
    # proj.info_storage()

    proj.learn(in_key_target = "best",
               in_keys_features = ["events", "estimate"], do_exclude = True,
               do_immediate_responses = False,
               in_strategy = strategy,
               in_tags_allocation = ["size_category", ("context", aml.AllocationMethod.LEAVE_ONE_OUT)])