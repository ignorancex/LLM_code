# ///////////// Copyright 2023 Vishnu Boddeti. All rights reserved. /////////////
# //
# //   Project     : Capacity of Generative Face Models
# //   File        : main.py
# //   Description : Main file to run capacity estimation for all datasets
# //
# //   Created On: 08/20/2023
# //   Created By: Vishnu Boddeti <mailto:vishnu@msu.edu>
# ////////////////////////////////////////////////////////////////////////////

from constants import *

import sys
import config
import traceback
from prettytable import PrettyTable

from capacity_estimation import CapacityEstimator
from utils import get_raw_values
from plotting import *

def main():
    # parse the arguments
    args = config.parse_args()

    for _, ref_dataset in enumerate(REF_DATASETS):
        for _, face_model in enumerate(FACE_MODELS):
            pt = PrettyTable()
            pt_far = PrettyTable()
            for d_id, dataset in enumerate(DATASETS):
                args.dataset = dataset
                args.face_model = face_model
                args.ref_dataset = ref_dataset

                if dataset.lower() != "dcface":
                    capacity = CapacityEstimator(args, gen_type='unconditional')
                    capacity.get_capacity(write_to_pkl_file=args.save_results)
                    capacity.get_capacity_gender(write_to_pkl_file=args.save_results)
                    capacity.get_capacity_age(write_to_pkl_file=args.save_results)
                else:
                    capacity = CapacityEstimator(args, gen_type='conditional')
                    capacity.get_capacity(write_to_pkl_file=args.save_results)
                    capacity.get_capacity_gender(write_to_pkl_file=args.save_results)

                plot_all_results(face_model, ref_dataset, dataset)
                cos_delta_, cap_, fars_, cap_at_fars_ = get_raw_values(dataset, ref_dataset,
                                                                       face_model, as_string=True)
                if d_id == 0:
                    pt.add_column("cos \u03B4", cos_delta_)
                    pt_far.add_column("FAR", fars_)
                pt.add_column(DATASET_LABELS[d_id], cap_)
                pt_far.add_column(DATASET_LABELS[d_id], cap_at_fars_)

            print(pt)
            print(pt_far)
    
    # plot all results together
    plot_joint_capacity()
    plot_joint_capacity_age()
    plot_joint_capacity_gender()
    
    # ablation study
    plot_same_dataset_capacity()
    plot_same_facemodel_capacity()
    plot_tabular_data_at_cos_similarity()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)