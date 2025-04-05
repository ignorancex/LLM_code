#!/usr/bin/env python

import os
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter

from randomised_filtering.evaluation import (
    get_output_folders,
    get_conditional_sets_from_folder,
    intersect,
    intersect_all_sets_in_list,
    unify_all_sets_in_list,
)
from randomised_filtering.streamline_indices import write_list_of_streamline_indices


DESC = """
This script evaluates streamline votes across all experiment instances found
in a folder. It analyzes the intersection of the sets of fully "positive" (P)
or fully "negative" (N) streamlines and writes the results to a file.
Additionally can write indices of streamlines with only positive/negative
votes to a json file.
"""
EPILOG = ""


OUTPUT_FOLDER_NAME = "output"


def build_argparser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument(
        "--implausible",
        required=False,
        action="store_true",
        help="Use this keyword to intersect implausible/negative streamlines. "
        "(default: False, i.e. analyze plausible/positive streamlines)",
    )
    p.add_argument(
        "--ar_limit",
        required=False,
        help="Limit for the acceptance rate (percentage of positive votes) from rSIFT "
        "for a streamline to be considered in the intersections "
        "(minimum AR for plausible streamlines, maximal AR for implausible ones)"
        "(default: if '--implausible', 0; otherwise, 100)",
    )
    p.add_argument(
        "--output_name",
        required=False,
        help="Name of json containing plausible streamlines (without file ending).",
    )
    return p


def main():
    args = vars(build_argparser().parse_args())

    # get all relevant folders from the path (name must begin with OUTPUT_FOLDER_NAME)
    folders = get_output_folders(os.getcwd(), OUTPUT_FOLDER_NAME)

    if args.get("implausible"):
        intersect_plausible = False
        folders.reverse()
        output_terms = {"plausible/implausible": "implausible", "P/N": "N", "><=": "<="}
        print("Intersecting implausible streamlines")
        ar_limit = 0

    else:
        intersect_plausible = True
        output_terms = {"plausible/implausible": "plausible", "P/N": "P", "><=": ">="}
        print("Intersecting plausible streamlines")
        ar_limit = 100

    if args.get("ar_limit"):
        ar_limit = int(args["ar_limit"])

    # smallest subset size for plausible, biggest for implausible
    first_subset = folders.pop(0)

    print("limit AR", ar_limit)
    print("folders", folders)

    filepath = os.path.join(os.getcwd(), first_subset)

    # get indices of streamlines which fulfill limit percentage of votes
    return_args = get_conditional_sets_from_folder(
        filepath, get_positives=intersect_plausible, percentage=ar_limit
    )  # TODO: use flag for returning only certain arguments!
    streamlines_first_subset = return_args[0] if intersect_plausible else return_args[1]
    streamline_index = return_args[-1]

    len_first_streamline_set = len(streamlines_first_subset)

    # save results from all experiment instances in lists for later use
    # streamlines in streamlines_first_subset which fulfill condition for intersection
    #   in other experiment instances
    ar_limit_fulfilled = []

    # streamlines in streamlines_first_subset which were not in set that fulfills
    #   condition for intersection in other experiment instances
    #   (may be both because of getting a different vote
    #   or because not having been seen)
    ar_limit_not_fulfilled = []

    # streamlines in streamlines_first_subset which do not fulfill the condition
    #   for intersection in other experiment instances
    other_votes = []

    # streamlines in streamlines_first_subset which were not seen in other
    #   experiment instances
    not_seen = []

    # streamline index from all experiment instances
    s_i = [streamline_index]

    path_to_csv = (
        output_terms["plausible/implausible"]
        + "_"
        + str(first_subset)
        + "_"
        + str(ar_limit)
        + ".csv"
    )
    with open(path_to_csv, "w") as f:

        f.write(
            "# streamlines with {} {}%% {} votes in {}: {}\n\n".format(
                output_terms["><="],
                ar_limit,
                output_terms["P/N"],
                first_subset,
                len_first_streamline_set,
            )
        )
        f.write(
            "subset size;# streamlines in intersection;%% of streamlines in "
            "intersection;# streamlines not in intersection;"
            "# streamlines with N votes;# streamlines not seen\n"
        )

        # compare "strictest" set of plausible streamlines with those from the
        #   other experiment instances
        for folder in folders:
            print("Looking at folder", folder)

            # get sets of streamline indices from other experiment instance
            (
                streamline_p_tmp,
                streamline_n_tmp,
                streamline_not_seen_tmp,
                s_i_tmp,
            ) = get_conditional_sets_from_folder(
                folder, get_positives=intersect_plausible, percentage=ar_limit
            )

            print("Checking index set intersections\n")

            intersection_not_seen = streamlines_first_subset & streamline_not_seen_tmp
            not_seen.append([folder, intersection_not_seen])

            s_i.append(s_i_tmp)

            if intersect_plausible:
                intersect_with_1 = streamline_p_tmp
                intersect_with_2 = streamline_n_tmp
            else:
                intersect_with_1 = streamline_n_tmp
                intersect_with_2 = streamline_p_tmp
            (
                intersection_tmp,
                ar_limit_not_fulfilled_tmp,
                intersection_other_votes,
            ) = intersect(streamlines_first_subset, intersect_with_1, intersect_with_2)
            ar_limit_fulfilled.append(intersect_with_1)

            ar_limit_not_fulfilled.append([folder, set(ar_limit_not_fulfilled_tmp)])
            other_votes.append([folder, intersection_other_votes])

            f.write(
                "{};{};{}%%;{};{};{}\n".format(
                    folder,
                    len(ar_limit_fulfilled),
                    round(
                        100
                        * len(intersection_tmp)
                        / (len_first_streamline_set - len(intersection_not_seen)),
                        2,
                    ),
                    len(ar_limit_not_fulfilled_tmp),
                    len(intersection_other_votes),
                    len(intersection_not_seen),
                )
            )

        # get streamlines which fulfilled the AR limit in each experiment instance
        intersect_all_subsets = streamlines_first_subset
        for i, elem in enumerate(ar_limit_fulfilled):
            intersect_all_subsets = intersect_all_subsets & elem
            f.write(
                "left after intersection with {}:; {}\n".format(
                    folders[i], len(intersect_all_subsets)
                )
            )

        f.write(
            "\nIntersection between {} streamlines from all sets: {}\n".format(
                output_terms["plausible/implausible"], len(intersect_all_subsets)
            )
        )

        # short analysis about the amounts of votes
        amounts_votes = [
            sum([len(s[x][0]) + len(s[x][1]) for s in s_i])
            for x in intersect_all_subsets
        ]
        amounts_votes = np.sort(np.array(amounts_votes))
        print("Least votes:", amounts_votes[0])
        print("Most votes:", amounts_votes[-1])
        print("Median:", np.median(amounts_votes))
        print("Mean:", np.mean(amounts_votes))

        f.write(
            "Intersection between all sets of streamlines not included: {}\n".format(
                len(intersect_all_sets_in_list(ar_limit_not_fulfilled))
            )
        )
        f.write(
            "Intersection between all sets of streamlines negative: {}\n".format(
                len(intersect_all_sets_in_list(other_votes))
            )
        )
        f.write(
            "Unification between all sets of streamlines negative: {}\n".format(
                len(unify_all_sets_in_list(other_votes))
            )
        )
        f.write(
            "Intersection between all sets of streamlines not seen: {}\n".format(
                len(intersect_all_sets_in_list(not_seen))
            )
        )
        f.write(
            "Unification between all sets of streamlines not seen: {}\n".format(
                len(unify_all_sets_in_list(not_seen))
            )
        )

        # write streamline indices if output file path is given
        if args.get("output_name"):
            write_list_of_streamline_indices(
                args["output_name"] + ".json",
                list(intersect_all_subsets),
                os.path.join(os.getcwd(), "all.trk"),
            )


if __name__ == "__main__":
    main()
