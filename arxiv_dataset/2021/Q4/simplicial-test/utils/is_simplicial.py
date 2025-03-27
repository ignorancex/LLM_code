#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simplicial_test -- a python module to realize simplicial degree-size sequences
#
# Copyright (C) 2020-2021 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import click

try:
    from simplicial_test import Test, compute_joint_seq, if_facets_simplicial
except ModuleNotFoundError:
    import sys

    sys.path[0] += "/.."
    from simplicial_test.Test import Test, compute_joint_seq, if_facets_simplicial


@click.command()
@click.option('-k', '--degree_seq_file', 'degree_sequence', type=click.File('r'), help='Path to degree sequence file.')
@click.option('-s', '--size_seq_file', 'size_sequence', type=click.File('r'), help='Path to size sequence file.')
@click.option('-c', '--cutoff', 'cutoff', type=click.INT, help='Cutoff (max. steps) to the search.', default=1e5)
@click.option('-w', '--width', 'width', type=click.INT, help='Search width (see the docs).', default=1e2)
@click.option('-d', '--depth', 'depth', type=click.INT, help='Backtrack depth (see the docs).', default=1e10)
@click.option('--verbose', is_flag=True, help='Turn on the verbose mode.')
def is_simplicial(degree_sequence, size_sequence, cutoff, width, depth, verbose):
    degree_sequence = list(map(int, degree_sequence.read().replace("\n", "").split(" ")))
    size_sequence = list(map(int, size_sequence.read().replace("\n", "").split(" ")))
    st = Test(degree_sequence, size_sequence, cutoff=cutoff, width=width, depth=depth, verbose=verbose)
    is_simplicial, facets = st.is_simplicial()
    if is_simplicial is True:
        joint_seqs = compute_joint_seq(facets)
        assert if_facets_simplicial(facets) is True
        assert joint_seqs[0] == sorted(degree_sequence, reverse=True)
        assert joint_seqs[1] == sorted(size_sequence, reverse=True)
        print(f"Yes, the degree-size sequence is simplicial. \nThe complex is: {facets}")
    else:
        print("No, the degree-size sequence cannot realize a simplicial complex.")


if __name__ == "__main__":
    is_simplicial()
