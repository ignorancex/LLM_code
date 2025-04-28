#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import copy
import cross
import numpy as np
from math import sqrt
import points_generators
from common import distance, same_points, same


def set_dist_threshold(radius, k):
  return radius * sqrt(3) / 5 / (k + 1) ** 2

def gen_initial_points():
    points = []

    points.extend(points_generators.gen_points51())

    # points.extend(points_generators.gen_points47())
    # points.extend(points_generators.gen_points0())
    # points.extend(points_generators.gen_points46())

    # # no nz4flow, has nz5flow, 30 points, Petersen graph
    # points.extend(points_generators.gen_points7())
    # points.extend(points_generators.gen_points8())
    # points.extend(points_generators.gen_points14())
    # points.extend(points_generators.gen_points31())
    # points.extend(points_generators.gen_points34())
    # points.extend(points_generators.gen_points35())
    # points.extend(points_generators.gen_points39())
    # points.extend(points_generators.gen_points41())
    # points.extend(points_generators.gen_points42())
    # # artificial example
    # points.extend(points_generators.gen_points9())

    # # no nz3flow, has nz4flow, 12 points, K4 graph
    # points.extend(points_generators.gen_points4())
    # points.extend(points_generators.gen_point15())
    return points

######################################################################################

# set all points to the same radius
def set_radius(points, radius):
    for i in range(len(points)):
        coef = radius / distance(points[i], [0.0, 0.0, 0.0])
        for j in range(len(points[i])):
            points[i][j] = points[i][j] * coef
    return points


# remove same points
def uniq_points(points):
    print("before uniq points:", len(points), file=sys.stderr)
    for_remove = set()
    for i in range(len(points)):
        if i not in for_remove:
            for j in range(i + 1, len(points)):
                if j not in for_remove and same_points(points[i], points[j]):
                    for_remove.add(j)
    filtered_points = []
    for i in range(len(points)):
        if i not in for_remove:
            filtered_points.append(points[i])
    print("after uniq points:", len(filtered_points), file=sys.stderr)
    return filtered_points


def add_middle_points(points, radius):
    dist = radius * sqrt(3) / 4
    new_points = []
    for i in range(len(points)):
        for j in range(i, len(points)):
            if np.random.random() < 0.3:
              continue
            if (distance(points[i], points[j]) < dist):
                p3 = [points[i][0] + points[j][0], points[i][1] + points[j][1], points[i][2] + points[j][2]]
                if not same_points(p3, [0.0, 0.0, 0.0]):
                    new_points.append(p3)
    return new_points

# add opposite points, if they are not here already
def add_opposite_points(points):
    opposite_points = dict()
    i = 0
    while i < len(points):
        if i not in opposite_points:
            for j in range(i + 1, len(points)):
                if same_points(points[i], [-x for x in points[j]]):
                    opposite_points[i] = j
                    opposite_points[j] = i
                    break
            if i not in opposite_points:
                j = len(points)
                points.append([-x for x in points[i]])
                opposite_points[i] = j
                opposite_points[j] = i
        i += 1
    return points, opposite_points


# setup initial great circles
def gen_initial_great_circles(points, opposite_points, radius, k):
    dist = set_dist_threshold(radius, k)
    edges = set()
    for i in range(len(points)):
        #print(i, file=sys.stderr)
        for j in range(i + 1, len(points)):
            if np.random.random() < 0.3:
              continue

            if (opposite_points[i], opposite_points[j]) not in edges and (opposite_points[j], opposite_points[i]) not in edges \
                    and distance(points[i], points[j]) < dist:
                if opposite_points[i] != j:
                    cur_normal = np.cross(points[i], points[j])
                    cur_normal /= distance(cur_normal, [0.0, 0.0, 0.0])
                    found_same_edge = False
                    for e in edges:
                        prev_normal = np.cross(points[e[0]], points[e[1]])
                        prev_normal /= distance(prev_normal, [0.0, 0.0, 0.0])
                        if same_points(cur_normal, prev_normal) or same_points(cur_normal, [-x for x in prev_normal]):
                            found_same_edge = True
                            break
                    if not found_same_edge:
                        edges.add((i, j))
    return list(edges)


# find all points of intersections of great circles; and all points on every great circle
def intersect_great_circles(points, opposite_points, edges):
    print("points:", len(points), "edges:", len(edges), file=sys.stderr)
    all_points = copy.deepcopy(points)
    all_opposite_points = copy.deepcopy(opposite_points)
    circles_to_points = []
    for i in range(len(edges)):
        print(i, file=sys.stderr)
        cur_points = set()
        for j in range(len(edges)):
            if j != i:
                points_in_cross = cross.cross(points, edges[i], edges[j])
                found = False
                idx_found = -1
                for idx in range(len(all_points)):
                    if same_points(points_in_cross[0], all_points[idx]):
                        found = True
                        idx_found = idx
                        break
                idx1, idx2 = -1, -1
                if not found:
                    idx1 = len(all_points)
                    all_points.append(points_in_cross[0])
                    idx2 = len(all_points)
                    all_points.append(points_in_cross[1])
                    all_opposite_points[idx1] = idx2
                    all_opposite_points[idx2] = idx1
                else:
                    idx1 = idx_found
                    idx2 = all_opposite_points[idx1]
                cur_points.add(idx1)
                cur_points.add(idx2)
        circles_to_points.append(list(cur_points))
    return all_points, all_opposite_points, circles_to_points


# find all great triangles aka trinities
def find_great_triangles(all_points, radius, edges, circles_to_points):
    trinities = set()
    triangle_distance = radius * sqrt(3)
    for e in range(len(edges)):
        circle = circles_to_points[e]
        for i in range(len(circle)):
            for j in range(i + 1, len(circle)):
                d1 = distance(all_points[circle[i]], all_points[circle[j]])
                if triangle_distance > 0 and not same(d1, triangle_distance):
                    continue
                for k in range(j + 1, len(circle)):
                    d2 = distance(all_points[circle[i]], all_points[circle[k]])
                    d3 = distance(all_points[circle[j]], all_points[circle[k]])
                    if same(d1, d2) and same(d1, d3) and same(d2, d3):
                        numbers = [circle[i], circle[j], circle[k]]
                        numbers.sort()
                        trinities.add(tuple(numbers))
                        break
    return trinities


def find_trinities_from_points(points, opposite_points, radius):
    pairs = dict()
    triangle_distance = radius * sqrt(3)
    for i in range(len(points)):
        pairs[i] = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if  opposite_points[j] not in pairs[opposite_points[i]] and opposite_points[i] not in pairs[opposite_points[j]]:
                d = distance(points[i], points[j])
                if same(d, triangle_distance):
                    pairs[i].add(j)
                    opp = [opposite_points[i], opposite_points[j]]
                    opp.sort()
                    pairs[opp[0]].add(opp[1])
    for i in range(len(points)):
        if not pairs[i]:
            pairs.pop(i)

    trinities = set()
    for i in pairs:
        candidates = list(pairs[i])
        candidates.sort()
        for j in range(len(candidates)):
            for k in range(j + 1, len(candidates)):
                if candidates[j] in pairs and candidates[k] in pairs[candidates[j]]:
                    trinities.add((i, candidates[j], candidates[k]))
                    opp = [opposite_points[i], opposite_points[candidates[j]], opposite_points[candidates[k]]]
                    opp.sort()
                    trinities.add(tuple(opp))
    return trinities


# filter out uninteresting trinities
def filter_trinities(trinities):
    while True:
        prev_size = len(trinities)
        counts = dict()
        for t in trinities:
            for i in range(len(t)):
                if t[i] not in counts:
                    counts[t[i]] = 0
                counts[t[i]] += 1
        new_trinities = set()
        for t in trinities:
            ones = 0
            for i in range(len(t)):
                if counts[t[i]] == 1:
                    ones += 1
            if ones < 1: # probably don't need a fix, but previously FIXME should be "if ones < 2:"
                new_trinities.add(t)
        trinities = copy.deepcopy(new_trinities)
        if len(trinities) == prev_size:
            break
    return trinities


# map values to smaller
def map_to_smaller_indices(all_points, all_opposite_points, trinities):
    points = []
    smaller_indices = dict()
    cur_idx = 0
    for t in trinities:
        for i in range(len(t)):
            if t[i] not in smaller_indices:
                smaller_indices[t[i]] = cur_idx
                points.append(all_points[t[i]])
                cur_idx += 1

    opposite_points = dict()
    for idx in smaller_indices:
        opposite_points[smaller_indices[idx]] = smaller_indices[all_opposite_points[idx]]

    new_trinities = set()
    for t in trinities:
        new_trinities.add((smaller_indices[t[0]], smaller_indices[t[1]], smaller_indices[t[2]]))

    return points, opposite_points, new_trinities


def print_points(points, opposite_points, radius, trinities):
    if not(len(trinities)):
        print("No triangles!", file=sys.stderr)
    else:
        print("Found some triangles!", file=sys.stderr)

        trinities_by_idx = dict()
        for t in trinities:
            for v in t:
                if v not in trinities_by_idx:
                    trinities_by_idx[v] = []
                not_vs = []
                for x in t:
                    if x != v:
                        not_vs.append(x)
                trinities_by_idx[v].append(tuple(not_vs))

        colors = dict()
        for i in opposite_points:
            colors[i] = 0
        for seed_vertex in opposite_points:
            if not colors[seed_vertex]:
                flooded = []
                to_check = set([seed_vertex])
                while to_check:
                    cur_val = to_check.pop()
                    if cur_val not in flooded:
                        flooded.append(cur_val)
                    for t in trinities_by_idx[cur_val]:
                        for v in t:
                            if v not in flooded:
                                flooded.append(v)
                                to_check.add(v)
                print("seed:", seed_vertex, "flooded:", len(flooded), sep='\t', file=sys.stderr)
                for v in flooded:
                    colors[v] = 1

        print("radius", radius, sep='\t', file=sys.stdout)
        for i in range(len(points)):
            print("point", i, points[i][0], points[i][1], points[i][2], sep='\t', file=sys.stdout)
        for t in trinities:
            print("trinity", t[0], t[1], t[2], sep='\t', file=sys.stdout)
        for d in opposite_points:
            print("opposite", d, opposite_points[d], sep='\t', file=sys.stdout)

######################################################################################

def main():
    points = gen_initial_points()
    # radius = 3
    radius = (1+5**0.5)/2
    points = set_radius(points, radius)
    points = uniq_points(points)
    #points = points_generators.rotate1(points)
    #points = uniq_points(points)

    for k in range(2):
        if k:
            points = add_middle_points(points, radius)
            points = set_radius(points, radius)
            points = uniq_points(points)
        points, opposite_points = add_opposite_points(points)
        print("points:", len(points), file=sys.stderr)

        approach = 2

        if approach == 1:
            edges = gen_initial_great_circles(points, opposite_points, radius, k)
            all_points, all_opposite_points, circles_to_points = intersect_great_circles(points, opposite_points, edges)
            print('all points', len(all_points))
            trinities = find_trinities_from_points(all_points, all_opposite_points, radius)
            #trinities = find_great_triangles(all_points, radius, edges, circles_to_points)
        elif approach == 2:
            all_points = points
            all_opposite_points = opposite_points
            trinities = find_trinities_from_points(points, opposite_points, radius)

        trinities = filter_trinities(trinities)
        points, opposite_points, trinities = map_to_smaller_indices(all_points, all_opposite_points, trinities)
        if approach == 2:
          break

    print_points(points, opposite_points, radius, trinities)
    print("len(trinities)", len(trinities))

if __name__ == "__main__":
    main()
