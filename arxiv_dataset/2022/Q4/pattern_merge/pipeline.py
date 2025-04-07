from agglomerative import agglomerative_clustering
from metrics import *

datasets = ["datasets/chess.txt", "datasets/mushroom.txt", "datasets/connect.txt","datasets/pumsb.txt", "datasets/accidents.txt"]

for file in datasets:
    print("Current file: " + file)
    f = open(file, "r")
    nr_lines = 0
    variables = []

    # Read file
    for line in f:
        curr_transaction = line.split(" ")
        for value in curr_transaction:
            processed_value = value.replace("\n", "")
            if processed_value not in variables:
                variables.append(processed_value)
        nr_lines += 1
    f.seek(0)
    # Create dataframe
    df = pd.DataFrame(data=np.zeros((nr_lines, len(variables)), dtype=int), columns=variables)
    curr_row_pointer = 0
    for line in f:
        curr_transaction = line.split(" ")
        for variable in curr_transaction:
            processed_value = variable.replace("\n", "")
            df.at[curr_row_pointer, processed_value] = 1
        curr_row_pointer += 1

    df = df.drop(columns=[""])

    ones = 0
    for col in df.columns:
        ones += len(df[df[col] == 1])

    print(ones)

    number_of_variables = [4,8,12,16,20]
    #number_of_variables = [6,12,18,24,30,36]
    #number_of_variables = [8,16,24,32,40]
    max_per_cluster = 4
    #max_per_cluster = 6
    #max_per_cluster = 8

    # Statistical significance
    total_entries = df.shape[0] * df.shape[1]
    p = ones/total_entries
    n = len(df[df.columns[0]])
    p = pow(p, max_per_cluster)
    min_support = binom.ppf(q=0.05, n=n, p=p)/n

    print("Minimum support : "+str(min_support))

    for N in number_of_variables:
        print("#########################################################################################")
        print("Number of variables: " + str(N))

        print()
        print()
        print("#########################################################################################")
        print("#########################################################################################")
        print("Similar")
        print("#########################################################################################")

        similarity_m_times = []
        hierarchical_times = []
        apriori_clusters_times = []
        merging_times = []
        intersections = []
        variables_subsets = []
        variables_per_pattern = []
        observations_per_pattern = []

        #hierarchichal clustering properties
        patterns_discovered = []

        stored_dfs = []
        for counter in range(10):
            columns_to_use = random.sample(list(df.columns), N)
            copy_df = df[columns_to_use]
            stored_dfs.append(copy_df)

            start = time.time_ns()
            copy_df = copy_df.replace(0, "?")
            similarity_m = construct_distance_matrix(copy_df, "constant")

            s2 = time.time_ns()
            similarity_m_times.append((s2 - start) / 1000000000)
            print("similarity matrix: " + str((s2 - start) / 1000000000) + " seconds")

            aglo_start = time.time_ns()

            clusters = agglomerative_clustering(similarity_m, max_per_cluster)
            print(clusters)


            aglo_end = time.time_ns()
            hierarchical_times.append((aglo_end - aglo_start) / 1000000000)
            print("Hierarchical clustering: " + str((aglo_end - aglo_start) / 1000000000) + " seconds")

            s1 = time.time_ns()
            clusters_apriori = []
            print("Number of clusters : " + str(len(clusters.keys())))

            for cluster in clusters.keys():
                columns_to_use = []
                for value in clusters[cluster]:
                    columns_to_use.append(copy_df.columns[value])
                df_copy = copy_df[columns_to_use].copy()
                y = runApriori(data_iter=df_copy, class_vector=0, minSupport=min_support, minLift=1.3)
                y = merge_patterns([y], nr_lines, min_support)
                clusters_apriori.append(y)

            end = time.time_ns()
            apriori_clusters_times.append((end - s1) / 1000000000)
            print("Apriori with clusters: " + str((end - s1) / 1000000000) + " seconds")
            print("#########################################################################################")
            print("Intermediate analysis")
            nr_patterns_bef = []
            size_obs_bef = []
            size_vars = []
            for cluster in clusters_apriori:
                nr_patterns_bef.append(len(cluster))
                variables_p = []
                obs_p = []
                for pattern in cluster:
                    variables_p.append(len(pattern["columns"]))
                    obs_p.append(len(pattern["indexes"]))
                size_obs_bef.append(round((sum(obs_p)/(len(cluster) if len(cluster) > 0 else 1)), 0))
                size_vars.append(round(sum(variables_p)/(len(cluster) if len(cluster) > 0 else 1), 2))

            observations_per_pattern.append(sum(size_obs_bef)/len(clusters_apriori))
            variables_per_pattern.append(sum(size_vars)/len(clusters_apriori))
            print("Number of patterns per cluster: "+ str(nr_patterns_bef))
            print("Average number of observations per cluster: "+ str(size_obs_bef))
            print("Average number of variables per cluster: "+ str(size_vars))
            print("End of intermediate analysis")
            print("#########################################################################################")
            start_merge = time.time_ns()
            intermediate_intersections = []
            intermediate_variables = []
            result = merge_patterns_a(clusters_apriori, nr_lines, min_support, intermediate_intersections, intermediate_variables)
            intersections.append(sum(intermediate_intersections))
            variables_subsets.append(sum(intermediate_variables))
            end_merge = time.time_ns()
            merging_times.append((end_merge - start_merge) / 1000000000)

            patterns_discovered.append(len(result))
            print("Merging with clusters: " + str((end_merge - start_merge) / 1000000000) + " seconds")
            print("Combined time: " + str((end_merge - start) / 1000000000) + " seconds")

        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")
        print("For "+str(N)+" variables the time were:")
        print("Similarity Matrix")
        print(str(similarity_m_times))
        print("Hierarchical clustering")
        print(str(hierarchical_times))
        print("Apriori with clusters")
        print(str(apriori_clusters_times))
        print("Merging of patterns")
        print(str(merging_times))

        print("Number of intersections")
        print(str(intersections))
        print("Number of variables tested for subsets")
        print(str(variables_subsets))
        print("Number of observations per pattern")
        print(str(observations_per_pattern))
        print("Number of variables per pattern")
        print(str(variables_per_pattern))

        print("Number of patterns discovered")
        print(str(patterns_discovered))
        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")

        print("#########################################################################################")
        print("Dissimilar")
        print("#########################################################################################")
        similarity_m_times = []
        hierarchical_times = []
        apriori_clusters_times = []
        merging_times = []
        intersections = []
        variables_subsets = []
        variables_per_pattern = []
        observations_per_pattern = []


        #hierarchichal clustering properties
        patterns_discovered = []

        for s_df in stored_dfs:
            copy_df = s_df

            start = time.time_ns()
            copy_df = copy_df.replace(0, "?")
            similarity_m = construct_distance_matrix(copy_df, "constant")

            for i in range(0, len(similarity_m)):
                for j in range(0, len(similarity_m)):
                    if i == j:
                        similarity_m[i][j] = 0
                    else:
                        similarity_m[i][j] = 1 - similarity_m[i][j]

            s2 = time.time_ns()
            similarity_m_times.append((s2 - start) / 1000000000)
            print("similarity matrix: " + str((s2 - start) / 1000000000) + " seconds")

            aglo_start = time.time_ns()

            clusters = agglomerative_clustering(similarity_m, max_per_cluster)
            print(clusters)

            aglo_end = time.time_ns()
            hierarchical_times.append((aglo_end - aglo_start) / 1000000000)
            print("Hierarchical clustering: " + str((aglo_end - aglo_start) / 1000000000) + " seconds")

            s1 = time.time_ns()
            clusters_apriori = []
            print("Number of clusters : " + str(len(clusters.keys())))
            for cluster in clusters.keys():
                columns_to_use = []
                for value in clusters[cluster]:
                    columns_to_use.append(copy_df.columns[value])
                df_copy = copy_df[columns_to_use].copy()
                y = runApriori(data_iter=df_copy, class_vector=0, minSupport=min_support, minLift=1.3)
                y = merge_patterns([y], nr_lines, min_support)
                clusters_apriori.append(y)

                #print(len(y))
            end = time.time_ns()
            apriori_clusters_times.append((end - s1) / 1000000000)
            print("Apriori with clusters: " + str((end - s1) / 1000000000) + " seconds")

            print("Intermediate analysis")
            nr_patterns_bef = []
            size_obs_bef = []
            size_vars = []
            for cluster in clusters_apriori:
                nr_patterns_bef.append(len(cluster))
                variables_p = []
                obs_p = []
                for pattern in cluster:
                    variables_p.append(len(pattern["columns"]))
                    obs_p.append(len(pattern["indexes"]))
                size_obs_bef.append(round((sum(obs_p)/(len(cluster) if len(cluster) > 0 else 1)), 0))
                size_vars.append(round(sum(variables_p)/(len(cluster) if len(cluster) > 0 else 1), 2))


            observations_per_pattern.append(sum(size_obs_bef)/len(clusters_apriori))
            variables_per_pattern.append(sum(size_vars)/len(clusters_apriori))
            print("Number of patterns per cluster: "+ str(nr_patterns_bef))
            print("Average number of observations per cluster: "+ str(size_obs_bef))
            print("Average number of variables per cluster: "+ str(size_vars))
            print("End of intermediate analysis")


            start_merge = time.time_ns()
            intermediate_intersections = []
            intermediate_variables = []
            result = merge_patterns_a(clusters_apriori, nr_lines, min_support, intermediate_intersections, intermediate_variables)
            intersections.append(sum(intermediate_intersections))
            variables_subsets.append(sum(intermediate_variables))
            end_merge = time.time_ns()
            merging_times.append((end_merge - start_merge) / 1000000000)

            patterns_discovered.append(len(result))
            print("Merging with clusters: " + str((end_merge - start_merge) / 1000000000) + " seconds")
            print("Combined time: " + str((end_merge - start) / 1000000000) + " seconds")

        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")
        print("For "+str(N)+" variables the time were:")
        print("Similarity Matrix")
        print(str(similarity_m_times))
        print("Hierarchical clustering")
        print(str(hierarchical_times))
        print("Apriori with clusters")
        print(str(apriori_clusters_times))
        print("Merging of patterns")
        print(str(merging_times))

        print("Number of intersections")
        print(str(intersections))
        print("Number of variables tested for subsets")
        print(str(variables_subsets))
        print("Number of observations per pattern")
        print(str(observations_per_pattern))
        print("Number of variables per pattern")
        print(str(variables_per_pattern))

        print("Number of patterns discovered")
        print(str(patterns_discovered))
        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")

        print("#########################################################################################")
        print("Random")
        print("#########################################################################################")

        similarity_m_times = []
        hierarchical_times = []
        apriori_clusters_times = []
        merging_times = []
        intersections = []
        variables_subsets = []
        variables_per_pattern = []
        observations_per_pattern = []


        #hierarchichal clustering properties
        patterns_discovered = []

        for s_df in stored_dfs:

            copy_df = s_df

            start = time.time_ns()
            copy_df = copy_df.replace(0, "?")
            similarity_m = construct_distance_matrix(copy_df, "constant")

            for i in range(0, len(similarity_m)):
                for j in range(0, len(similarity_m)):
                    if i == j:
                        similarity_m[i][j] = 0
                    else:
                        similarity_m[i][j] = 1 - similarity_m[i][j]

            s2 = time.time_ns()
            similarity_m_times.append((s2 - start) / 1000000000)
            print("similarity matrix: " + str((s2 - start) / 1000000000) + " seconds")

            aglo_start = time.time_ns()

            #clusters = agglomerative_clustering(similarity_m, 4)
            columns_to_use = random.sample(list(copy_df.columns), N)
            clusters = {}
            next = -1
            for i in range(N):
                if i % max_per_cluster == 0:
                    next += 1
                if next not in clusters.keys():
                    clusters[next] = [columns_to_use[i]]
                else:
                    clusters[next].append(columns_to_use[i])

            print(clusters)

            aglo_end = time.time_ns()
            hierarchical_times.append((aglo_end - aglo_start) / 1000000000)
            print("Hierarchical clustering: " + str((aglo_end - aglo_start) / 1000000000) + " seconds")

            s1 = time.time_ns()
            clusters_apriori = []
            print("Number of clusters : " + str(len(clusters.keys())))
            for cluster in clusters.keys():
                columns_to_use = clusters[cluster]
                df_copy = copy_df[columns_to_use].copy()
                y = runApriori(data_iter=df_copy, class_vector=0, minSupport=min_support, minLift=1.3)
                y = merge_patterns([y], nr_lines, min_support)
                clusters_apriori.append(y)

                #print(len(y))
            end = time.time_ns()
            apriori_clusters_times.append((end - s1) / 1000000000)
            print("Apriori with clusters: " + str((end - s1) / 1000000000) + " seconds")

            print("Intermediate analysis")
            nr_patterns_bef = []
            size_obs_bef = []
            size_vars = []
            for cluster in clusters_apriori:
                nr_patterns_bef.append(len(cluster))
                variables_p = []
                obs_p = []
                for pattern in cluster:
                    variables_p.append(len(pattern["columns"]))
                    obs_p.append(len(pattern["indexes"]))
                size_obs_bef.append(round((sum(obs_p)/(len(cluster) if len(cluster) > 0 else 1)), 0))
                size_vars.append(round(sum(variables_p)/(len(cluster) if len(cluster) > 0 else 1), 2))


            observations_per_pattern.append(sum(size_obs_bef)/len(clusters_apriori))
            variables_per_pattern.append(sum(size_vars)/len(clusters_apriori))
            print("Number of patterns per cluster: "+ str(nr_patterns_bef))
            print("Average number of observations per cluster: "+ str(size_obs_bef))
            print("Average number of variables per cluster: "+ str(size_vars))
            print("End of intermediate analysis")

            start_merge = time.time_ns()
            intermediate_intersections = []
            intermediate_variables = []
            result = merge_patterns_a(clusters_apriori, nr_lines, min_support, intermediate_intersections, intermediate_variables)
            intersections.append(sum(intermediate_intersections))
            variables_subsets.append(sum(intermediate_variables))
            end_merge = time.time_ns()
            merging_times.append((end_merge - start_merge) / 1000000000)

            patterns_discovered.append(len(result))
            print("Merging with clusters: " + str((end_merge - start_merge) / 1000000000) + " seconds")
            print("Combined time: " + str((end_merge - start) / 1000000000) + " seconds")

        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")
        print("For "+str(N)+" variables the time were:")
        print("Similarity Matrix")
        print(str(similarity_m_times))
        print("Hierarchical clustering")
        print(str(hierarchical_times))
        print("Apriori with clusters")
        print(str(apriori_clusters_times))
        print("Merging of patterns")
        print(str(merging_times))

        print("Number of intersections")
        print(str(intersections))
        print("Number of variables tested for subsets")
        print(str(variables_subsets))
        print("Number of observations per pattern")
        print(str(observations_per_pattern))
        print("Number of variables per pattern")
        print(str(variables_per_pattern))

        print("Number of patterns discovered")
        print(str(patterns_discovered))
        print("#########################################################################################")
        print("#########################################################################################")
        print("#########################################################################################")
