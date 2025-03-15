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


from simplicial_test import Test


def test_not_simplicial_1():
    degree_list, size_list = ([2], [2])
    st = Test(degree_list, size_list)
    assert st.is_simplicial()[0] is False


def test_not_simplicial_2():
    degree_list, size_list = ([3, 3, 3, 2, 1, 1, 1], [6, 6, 2])
    st = Test(degree_list, size_list)
    assert st.is_simplicial()[0] is False


def test_not_simplicial_3():
    degree_list, size_list = ([3, 3, 3, 2, 1, 1, 1, 1, 1, 1], [6, 6, 2, 1, 1, 1])
    st = Test(degree_list, size_list)
    assert st.is_simplicial()[0] is False


def test_not_simplicial_4():
    degree_list, size_list = ([6, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1], [7, 7, 3, 2, 2, 2])
    st = Test(degree_list, size_list)
    assert st.is_simplicial()[0] is False


def test_not_simplicial_5():
    degree_list, size_list = (
        [6, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 7, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1])
    st = Test(degree_list, size_list)
    assert st.is_simplicial()[0] is False


def test_not_simplicial_unequal_seq_sum():
    degree_list, size_list = ([3, 3, 3, 2, 1, 1], [6, 6, 2])
    st = Test(degree_list, size_list)
    assert st.is_simplicial()[0] is False


def test_cutoff():
    degree_list, size_list = ([12, 8, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    st = Test(degree_list, size_list, verbose=False, width=1e4, cutoff=3000)
    assert st.is_simplicial()[0] is False


def test_rollback_to_level_0():
    degree_list, size_list = ([12, 10, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    st = Test(degree_list, size_list, verbose=0)
    assert st.is_simplicial()[0] is False
