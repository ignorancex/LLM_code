
def average_link(active_set, sim_matrix, max_num_per_cluster, broken_down):
    g1 = 0
    g2 = 0
    curr_best = math.inf
    for key1 in active_set.keys():
        for key2 in active_set.keys():
            if key1 == key2:
                continue
            x = active_set[key1]
            y = active_set[key2]
            if len(x)+len(y) > max_num_per_cluster:
                continue
            skip = False
            for b in broken_down:
                if key1 in b.keys() and key2 in b.keys():
                    if set(b[key1]) == set(x) and set(b[key2]) == set(y):
                        skip = True
                        break
            if skip:
                continue
            sumation = 0
            for val1 in x:
                for val2 in y:
                    sumation += sim_matrix[val1][val2]
            sumation /= (len(x)*len(y))
            if sumation < curr_best:
                curr_best = sumation
                g1 = key1
                g2 = key2
    return g1, g2, curr_best


def agglomerative_clustering(sim_matrix, max_num_per_cluster):
    active_set = {}
    finalized_clusters = {}
    broken_down = []
    combined_clusters = []
    for n in range(len(sim_matrix)):
        active_set[n] = [n]

    while len(active_set.keys()) > 1:
        g1, g2, dist = average_link(active_set, sim_matrix, max_num_per_cluster, broken_down)
        if dist == math.inf:
            #break down last item formed
            inverted_list = combined_clusters[::-1]
            for c in inverted_list:
                skip = False
                for val in c.keys():
                    if val in finalized_clusters.keys():
                        skip = True
                        break
                if skip:
                    continue
                for val in c.keys():
                    if val in active_set.keys():
                        active_set.pop(val)
                    active_set[val] = c[val]
                list_keys = list(c.keys())
                broken_down.append({list_keys[0]: c[list_keys[0]], list_keys[1]: c[list_keys[1]]})

                for i in range(len(combined_clusters)):
                    val = combined_clusters[i]
                    if set(list(val.keys())) == set(list(c.keys())):
                        keys = list(val.keys())
                        if set(val[keys[0]]) == set(c[keys[0]]) and set(val[keys[1]]) == set(c[keys[1]]):
                            del combined_clusters[i]
                            break
                break

        else:
            #remove g1 and g2 from active and add
            combined = np.concatenate((active_set[g1], active_set[g2]))
            if len(combined) < max_num_per_cluster:
                combined_clusters.append({g1:active_set[g1],g2:active_set[g2]})
                active_set[g1] = combined
            else:
                active_set.pop(g1)
                finalized_clusters[g1] = combined

            active_set.pop(g2)

    return finalized_clusters
