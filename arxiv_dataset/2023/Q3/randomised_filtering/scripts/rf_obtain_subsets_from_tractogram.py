#!/usr/bin/env python

import os
import nibabel as nib
import json

from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent


DESC = """
Obtain a subset of the streamlines in a given tractogram file which correspond
to a given set of streamline indices provided in a json file. Used in Chapter 3.
"""
EPILOG = dedent(
    """
    example calls:

      {filename} <path_to_tractogram> <path_to_json_file> <path_to_output_file>
    """.format(
        filename=os.path.basename(__file__)
    )
)


def build_parser():
    p = ArgumentParser(
        description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter
    )
    p.add_argument("tractogram", help="Path to tractogram to be subsampled.")
    p.add_argument(
        "jsonlist",
        help="Path to json files containing streamline indices",
    )
    return p


if __name__ == "__main__":

    args = vars(build_parser().parse_args())
    jsonlist = args["jsonlist"].split("\n")
    print("Found following index jsons:", jsonlist)

    # load tractogram
    trk = nib.streamlines.load(args["tractogram"])  # type: nib.streamlines.TrkFile
    t = trk.tractogram.to_world()  # bring to RASmm
    print("Loaded tractogram")

    # load indices
    for jsonname in jsonlist:
        with open(jsonname, "r") as f:
            json_content = json.load(f)

        filenames = json_content["filenames"]

        s = "Loaded streamline indices:\n"
        for filename in filenames:
            s += "\t{file:<60} : {nb_streamlines:>8}\n".format(
                file=filename, nb_streamlines=len(json_content[filename])
            )
        s += " -> Using indices from {file_to_use}.".format(file_to_use=filenames[0])
        print(s)

        # create tractogram from subset of streamlines
        t_new = nib.streamlines.Tractogram(
            streamlines=t.streamlines[json_content[filenames[0]]],
            # NOTE: expected to be eye(4) at this point.
            affine_to_rasmm=t.affine_to_rasmm,
        )

        # write back subset
        nib.streamlines.save(
            # tractogram with corrected affine in rasmm
            #   (or 'world'mm; same as image data)
            tractogram=t_new,
            filename=jsonname[:-5] + ".trk",
            # header from given image data; defines the 'voxmm' space of the streamlines
            header=trk.header,
        )
