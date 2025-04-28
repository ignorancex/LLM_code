#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import numpy as np
from collections import Counter


# uses from outside: max_idx, flooded, color_order, opposite_points, trinities_by_idx, colors, flow_upper_bound
def find_colors(cur_idx):
    global max_cur
    max_cur = max(cur_idx, max_cur)

    if cur_idx > max_idx:
        return True
    cur_val = flooded[cur_idx]
    if colors[cur_val]:
        return find_colors(cur_idx + 1)
    for cur_color in color_order:
        if abs(cur_color) >= flow_upper_bound:
            break
        colors[cur_val], colors[opposite_points[cur_val]] = cur_color, -cur_color
        all_good = True
        also_colored = set()
        to_check = set([cur_val, opposite_points[cur_val]])
        checked_trinities = set()
        while all_good and to_check:
            val_for_check = to_check.pop()
            for t in trinities_by_idx[val_for_check]:
                if not t in checked_trinities: # probably no need for this structure
                    checked_trinities.add(t)
                    zeros_count = 0
                    zeros_vals = []
                    for v in t:
                        if not colors[v]:
                            zeros_count += 1
                            zeros_vals.append(v)
                    if zeros_count < 2:
                        flow = colors[val_for_check] + colors[t[0]] + colors[t[1]]
                        all_good = (zeros_count == 0 and flow == 0) or (zeros_count != 0 and flow != 0 and abs(flow) < flow_upper_bound)
                        if not all_good:
                            break
                        elif zeros_count == 1:
                            v = zeros_vals[0]
                            colors[v], colors[opposite_points[v]] = -flow, flow
                            also_colored.add(v)
                            also_colored.add(opposite_points[v])
                            to_check.add(v)
                            to_check.add(opposite_points[v])
        if all_good:
            #print("bound:", flow_upper_bound, "cur idx:", cur_idx, "also_colored:", len(also_colored), file=sys.stderr)
            if find_colors(cur_idx + 1):
                return True
        for idx in also_colored:
            colors[idx] = 0
    colors[cur_val], colors[opposite_points[cur_val]] = 0, 0
    return False


flow_upper_bound = 5
color_order = []
for i in range(1, flow_upper_bound):
    color_order.append(i)
    color_order.append(-i)

radius = -1
points = []
opposite_points = dict()
trinities = set()
f = open(sys.argv[1])
for line in f:
    values = line.strip().split('\t')
    if values[0] == 'point':
        points.append([float(v) for v in values[2:]])
    elif values[0] == 'radius':
        radius = float(values[1])
    elif values[0] == 'opposite':
        opposite_points[int(values[1])] = int(values[2])
    elif values[0] == 'trinity':
        trinities.add(tuple([int(v) for v in values[1:]]))
    else:
        print("WTF: problems with parsing input", file=sys.stderr)

colors = dict()
for i in opposite_points:
    colors[i] = 0
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
#print(trinities_by_idx, file=sys.stderr)

found4, found5 = True, True

for seed_vertex in opposite_points:
    if not colors[seed_vertex]:
        flooded = []
        should_check = set([seed_vertex])
        to_check = set([seed_vertex])
        checked = set()
        while to_check:
            cur_val = to_check.pop()
            if cur_val not in flooded:
                flooded.append(cur_val)
            checked.add(cur_val)
            done_1_flood = False
            np.random.shuffle(trinities_by_idx[cur_val])
            for t in trinities_by_idx[cur_val]:
                for v in t:
                    if v not in flooded:
                        if not done_1_flood:
                            flooded.append(v)
                            done_1_flood = True
                        to_check.add(v)
                        should_check.add(v)
            if not to_check:
                to_check = should_check - checked
        max_idx = len(flooded) - 1

        flow_upper_bound = 4
        max_cur = 0
        #cur_found4 = find_colors(0) # FIXME
        cur_found4 = False
        found4 = found4 and cur_found4
        print("seed:", seed_vertex, "flooded:", len(flooded), sep='\t', file=sys.stdout)
        if not cur_found4:
            print("\033[94mNo " + str(flow_upper_bound) + "-flow for this component!\033[0m", file=sys.stdout)
            print("max_cur:", max_cur, file=sys.stdout)
            print("radius", radius, sep='\t', file=sys.stderr)
            for v in should_check:
            #    #colors[v] = float('NaN')
                print("to_plot", points[v][0], points[v][1], points[v][2], sep='\t', file=sys.stderr)
        else:
            flows4 = dict()
            for v in flooded:
                flows4[v] = colors[v]
            #print("flow values:", flows4, file=sys.stdout)
            print("flow_values", flows4, sep='\t', file=sys.stderr)

        if not cur_found4:
            # TODO: remove copypasta
            flow_upper_bound = 5
            cur_found5 = find_colors(0)
            found5 = found5 and cur_found5
            if not cur_found5:
                print("\033[91mBRKNG! No " + str(flow_upper_bound) + "-flow for this component!\033[0m", file=sys.stdout)
                #print("radius", radius, sep='\t', file=sys.stderr)
                for v in flooded:
                    colors[v] = float('NaN')
                #    print("to_plot", points[v][0], points[v][1], points[v][2], sep='\t', file=sys.stderr)
            else:
                flows5 = dict()
                for v in should_check:
                    flows5[v] = colors[v]
                    print("color", colors[v], sep='\t', file=sys.stderr)
                if len(flooded) == len(points):
                    graph_triples = set()
                    graph_vertices = list()
                    for t in trinities:
                        print("trinity", t[0], t[1], t[2], sep='\t', file=sys.stderr)
                        v0 = min(opposite_points[t[0]], t[0])
                        v1 = min(opposite_points[t[1]], t[1])
                        v2 = min(opposite_points[t[2]], t[2])
                        graph_vertices.append(v0)
                        graph_vertices.append(v1)
                        graph_vertices.append(v2)
                        graph_triples.add(tuple(sorted([v0, v1, v2])))
                    for t in graph_triples:
                      print('graph triple', t)

                    cnt = Counter(graph_vertices)
                    cnt2 = {}
                    is_it_a_graph = True
                    for k in cnt:
                      cnt2[k] = cnt[k] // 2
                      if cnt2[k] != 2:
                        is_it_a_graph = False
                    print('graph multiplicities', cnt2)
                    print('is it a graph:', is_it_a_graph)
                    graph_vertices = set(graph_vertices)
                    print('graph edges', len(graph_vertices), 'idx:', graph_vertices)
                    print('graph "vertices"', len(graph_triples))
                    print('should be vertices:')
                    diffs = dict()
                    for t in graph_triples:
                      all_graph = True
                      multiples = []
                      for g in t:
                        if cnt2[g] != 2:
                          all_graph = False
                        multiples.append(cnt2[g])
                      if multiples == [4, 4, 4]:
                        for g in t:
                          if g not in diffs:
                            diffs[g] = 0
                          diffs[g] += 1
                      print(t, all_graph, multiples)
                      cnt3 = {}
                    is_it_a_graph2 = True
                    for g in cnt2:
                      if g not in diffs:
                        diffs[g] = 0
                      cnt3[g] = cnt2[g] - diffs[g]
                      if cnt3[g] != 2:
                        is_it_a_graph2 = False
                    print('graph multiplicities2', cnt3)
                    print('is it a graph2:', is_it_a_graph2)
                    if is_it_a_graph2:
                      pass
                #print("flow values:", flows5, file=sys.stdout)
                print("flow_values", flows5, sep='\t', file=sys.stderr)


print("Have found 5 flow:", found5, file=sys.stdout)

# double check
for t in trinities:
    color_sum = colors[t[0]] + colors[t[1]] + colors[t[2]]
    if color_sum:
        print("WTF trinity", t, colors[t[0]], colors[t[1]], colors[t[2]], file=sys.stdout)
        print("WTF trinity", t, colors[t[0]], colors[t[1]], colors[t[2]], file=sys.stderr)
for d in opposite_points:
    if colors[d] + colors[opposite_points[d]]:
        print("WTF opposite", (d, opposite_points[d]), colors[d], colors[opposite_points[d]], file=sys.stdout)
        print("WTF opposite", (d, opposite_points[d]), colors[d], colors[opposite_points[d]], file=sys.stderr)

if not found4:
    print("\033[94mno 4-flow!\033[0m", file=sys.stdout)
if not found5:
    print("\033[91mNO 5-FLOW!\033[0m", file=sys.stdout)
    print("\033[91mNO 5-FLOW!\033[0m", file=sys.stderr)
print('opposite points', opposite_points)
