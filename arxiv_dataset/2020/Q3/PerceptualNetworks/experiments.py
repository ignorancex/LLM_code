import numpy
import random
import itertools
import sys
import global_functions
import copy


class preliminary:

    def __init__(self):
        self.name = ""

    """ --- sexual selection and calculation of fitness (percept mutual information) ---"""

    def sexual_selection(self,fitness_aid):
        # select parents based on fitness:
        weights = []
        n_pos = 0
        s = 0
        for ga in fitness_aid:
            s += fitness_aid[ga][0]
            if fitness_aid[ga][0] > 0: n_pos += 1
        if n_pos < 2: return (0, 1)  # there must be at least two positice entries
        for ga in fitness_aid:
            weights.append(fitness_aid[ga][0] / s)
        ij = numpy.random.choice(range(len(fitness_aid)), p=weights, size=2, replace=False)
        # print("reproduce:", ij, weights)
        return ij

    def percept_MI(self,percepts, alpha, size_input, p_i0, print_flag):
        # calculate MI for different percepts
        p_flip = alpha / 100
        percept_hist = {}
        j = {}
        for i0 in itertools.product({0, 1}, repeat=size_input):
            for ia in itertools.product({0, 1}, repeat=size_input):
                distance = global_functions.hamming_distance(i0, ia, size_input)
                p_i0ia = p_i0 * (p_flip ** distance) * ((1 - p_flip) ** (size_input - distance))
                # determine "attractor number" for input ia
                source = global_functions.tuple_to_bin(i0)
                target = global_functions.tuple_to_bin(ia)
                j[int(source, 2)] = []
                att_number = []
                for tup in percepts[target]:
                    att_number.append(int(global_functions.tuple_to_bin(tup), 2))
                att_number = tuple(att_number)
                try:
                    percept_hist[int(source, 2), att_number] += p_i0ia
                except:
                    percept_hist[int(source, 2), att_number] = p_i0ia

        reduced_percept_hist = {}
        key_list = []
        for key in percept_hist:
            if key[1][0] not in key_list: key_list.append(key[1][0])
        if print_flag: print("Number of different attractors:", len(key_list), key_list)
        for key in percept_hist:
            new_key = key_list.index(key[1][0])
            reduced_percept_hist[key[0], new_key] = percept_hist[key]
            if print_flag: print(key[0], new_key, numpy.around(percept_hist[key], 8))
        mi = global_functions.hist_MI(reduced_percept_hist)
        if print_flag: print("Mutual information", numpy.around(mi, 3), ", maximum value:", size_input)
        return mi

    """ --- various subroutines used in the experiments --- """

    def signaling(self,model, inp, max_time_steps, i_a=True, print_flag=False):

        """ calculate time evolution of the network """
        numagents = model.numagents
        excitation = [model.agent[a].x for a in range(numagents)]
        excitation[0:len(inp)] = inp
        pattern = {}
        nt = 1
        attrac = False

        while nt <= max_time_steps and attrac == False:

            # update at each time step
            excitation0 = excitation
            messages = numpy.matmul(model.adjacency, numpy.array(excitation0))
            adj = model.adjacency.sum(axis=1)
            j = 0
            for a in range(numagents):
                model.agent[a].update(messages[a], adj[a], print_flag=False)
                # if numpy.random.random() < 0.05: model.agent[a].x = (model.agent[a].x + 1) % 2 #bit flip with probability 0.05
            excitation = [model.agent[a].x for a in range(numagents)]
            if print_flag: print(" " * 5, "Network update at step:", nt, excitation0, "--", tuple(messages), "->",
                                 excitation)

            # determine pattern at the end of run
            if i_a and excitation0 == excitation:  # reached a stationary state attractor
                attrac = True
                pattern[nt] = excitation0
                if print_flag:
                    print(" " * 5, "Stationary state reached at step:", nt)  # , excitation0,"->",excitation)
                break
            if i_a:  # check whether a cyclic attractor has been reached
                for key in pattern:
                    if excitation == pattern[key]:  # and nt%set_delay == key%set_delay:
                        pop_key = key
                        if print_flag:
                            print(" " * 5, "landed in non-stationary attracting set; discovered at step", nt)
                            print(" " * 5, "patterns starts at", pop_key)
                        attrac = True
                else:
                    pattern[nt] = excitation

            nt += 1

        if not attrac:
            # print("Warning: simulation not converged!")
            pattern[1] = excitation
        return (pattern, attrac)

    def infer_percepts(self,model, max_time_steps, size_input, print_flag=False, attractor_flag=False):

        """Determine possible perceptions and store results in dictionary xas"""
        xas = {}
        # initial_state = [0] * model.numagents  # list(numpy.random.randint(0, 2,numagents))
        for i in range(2 ** size_input):
            for a in range(model.numagents): model.agent[a].x = 0
            input = global_functions.int_to_bin(i, size_input)
            input_string = global_functions.tuple_to_bin(input)

            # check wether calculation has already been performed
            if len(xas) == (2 ** size_input):
                break  # quit the loop
            elif input_string in xas.keys():
                continue  # jump to next
            else:
                xas[input_string] = []
            if print_flag:
                print()
                print("---")
                print("Initial state:", [model.agent[a].x for a in range(model.numagents)], "receives input:", input)
            (pattern, attrac) = self.signaling(model, input, max_time_steps, i_a=True, print_flag=print_flag)
            # determine final state in pattern:
            n_steps = len(pattern)
            last_state = pattern[n_steps]

            # determine first occurences of final state (i.e. begin of cycle)

            first_occurance = n_steps
            for t in range(1, n_steps):
                if pattern[t] == last_state:
                    first_occurance = t
                    break

            for t in range(1, n_steps):
                source = global_functions.tuple_to_bin(pattern[t])
                target = global_functions.tuple_to_bin(pattern[t + 1])
                # print(t,source,target)
                model.trajectories[source] = target

            for t in pattern:
                if pattern[t] in xas[input_string]:
                    break
                elif t >= first_occurance or not attrac:
                    xas[input_string].append(pattern[t])
            if not attrac: xas[input_string] = [[0] * model.numagents]  # if not congerged: goto 0

        """ the dictionary xas contains all possible input-output transitions"""
        j = 0
        for key in xas:
            xas[key].sort()
            if len(xas[key]) < max_time_steps and attractor_flag:
                print("for input_string =", key, ": evolution landed in an attracting set of size", len(xas[key]))
        for key in xas:
            if attractor_flag: print(key, "->", xas[key])

        if print_flag:
            print("---")
            print()

        # for every input probability p(i_actual) = p_i0 * alpha ** H[i0;ia) * (1-alpha)** (size_input-H[i0;ia])
        # first check that all input-output pairs have been calculated:
        if len(xas) != 2 ** size_input:
            print("Warning: not all inputs have been determined, number inputs:", len(xas), "Exit.")
        return xas

    """--- various subroutines for re-wiring the network (experiment 2 only) """

    def delete_connection(self,adj,p_delete,offset=(),print_flag=False):

        indices = numpy.transpose(numpy.nonzero(adj))
        seeds = numpy.random.random(size=len(indices))
        # print(indices)
        j = 0
        for c in indices:
            # print(i,seeds[j],model.adjacency[tuple(i)])
            if 2*seeds[j] <= p_delete and (c[0] not in offset or c[1] not in offset): #every edge has the chance of breaking two times
                if print_flag: print("  ", "Remove edge", tuple(c), "from network")
                adj[tuple(c)] = 0
                if print_flag: print("  ", "Remove edge", tuple(reversed(c)), "from network")
                adj[tuple(reversed(c))] = 0
            j += 1
        return adj

    def add_node(self,adj,p_add,print_flag=False):
        #indices_existing = []
        #indices_free = []
        if numpy.random.random() < p_add:
            indices_existing = global_functions.agent_list_from_adj(adj)
            indices_free = [ia for ia in range(len(adj)) if ia not in indices_existing]
            #for ia in range(len(adj)):
            #    if numpy.sum(adj[ia]) != 0 or numpy.sum(adj[:, ia]) != 0: indices_existing.append(ia)
            #    else: indices_free.append(ia)
            #print(indices_existing,indices_free)
            if indices_free != []:
                p_existing = numpy.sum(adj,axis = 0)[indices_existing] #the rich get (probably) richer...
                if indices_existing == []: parent = 0
                else: parent = random.choices(indices_existing, weights=p_existing)[0]
                child = random.choice(indices_free)
                adj[child,parent] = 1
                adj[parent,child] = 1
                if print_flag: print("Add", child, "to", parent)
        return adj

    def new_connection(self,adj,p_new,print_flag=False):

        if numpy.random.random() < p_new:
            """ make one new connection """
            indices = global_functions.agent_list_from_adj(adj)
            source,target = 0,0
            while source == target: source,target = random.choices(indices,k=2)
            adj[source, target] = 1
            adj[target, source] = 1
            if print_flag: print("New connection from", source, "to", target)
        return adj

    def random_rewire(self,adj,p_flip,print_flag=False):

        #determine indices of existing agents
        indices = global_functions.agent_list_from_adj(adj)

        #determine indicies of mutations
        numagents = len(indices)
        indices2 = numpy.random.randint(0,len(adj),numagents)
        indices3 = numpy.random.randint(0,len(adj),numagents)
        seeds = numpy.random.rand(numagents)
        #print(indices,numagents,indices2)

        #for every existing agent, there can be one rewire
        for n in range(numagents):
            source = indices[n]
            target = indices2[n]
            target2 = indices3[n]
            #print(target,source,adj)
            if seeds[n] < p_flip and target != source and target2 != source :
                a0 = copy.copy(adj[target,source])
                adj[target, source] = adj[target2, source]
                adj[target2,source] = a0
                if print_flag: print("mutation of adjacency", target,source, "->", target2,source)
                #print(adj)
        return adj

    def mutate_connectivity(self,adjacency,p_flip,print_flag=False):

        """mutate exactly one connection in network"""
        numagents = len(adjacency)
        index = numpy.random.randint(0,numagents,2)
        if numpy.random.random() < p_flip:
            while index[0] == index[1]: index = numpy.random.randint(0,numagents,2) #diagonal elements have to be zero
            adjacency[index[1],index[0]] = (adjacency[index[1],index[0]] + 1)%2
            adjacency[index[0],index[1]] = adjacency[index[1],index[0]]
            if print_flag: print("mutation of adjacency", index[1],index[0], "to", adjacency[index[1],index[0]])
        return adjacency

    def change_network(self,model,p_add,p_death,p_mutate,p_losing_touch,p_new_connection,cost=1,print_flag=False):

        """ this subroutine introduces various changes in size and connectivity of the network """

        #1. every agent has a finite life-time
        seed = numpy.random.random(size=model.numagents)
        for ia in reversed(range(model.numagents)):  # remove agents from the network
            if seed[ia] <= p_death:
                # remove agent with model0.agent[a1].agentID from model
                # print(model.adjacency)
                if model.agent[ia].in_node:
                    continue
                else:
                    if print_flag: print("  ", "Remove agent", model.agent[ia].agentID)
                    del model.agent[ia]
                    model.numagents -= 1
                    model.adjacency = numpy.delete(model.adjacency, ia, 0)
                    model.adjacency = numpy.delete(model.adjacency, ia, 1)
                # print(model.adjacency)

        #2. add new element to network
        if numpy.random.random() <= p_add:
            # determine the parent wigthed by connections (highly connected entities are more likely to have offspring)
            probs = numpy.sum(model.adjacency, axis=0)
            norm = numpy.linalg.norm(probs, ord=1)
            # print(probs,norm)
            probs = probs / norm if norm != 0 else probs
            parent = random.choices(range(model.numagents), weights=probs, k=1)[0]
            # add agent to network
            if print_flag: print("  ", "Add agent to parent", parent)
            model.add_agent(parent, 0.01, True)
            # construct new adjacency
            z = numpy.zeros((model.numagents, model.numagents), dtype=int)
            z[0:len(model.adjacency), 0:len(model.adjacency)] = model.adjacency
            z[parent, model.numagents - 1] = 1
            z[model.numagents - 1, parent] = 1
            model.adjacency = z
            # print("new adjacency:", model.adjacency)

        #3. randomly change the connectivity of the network
        model.adjacency = self.mutate_connectivity(model.adjacency, p_mutate, print_flag)

        #4. every connection has finite lifetime
        indices = numpy.transpose(numpy.nonzero(model.adjacency))
        seeds = numpy.random.random(size=len(indices))
        # print(indices)
        j = 0
        for i in indices:
            # print(i,seeds[j],model.adjacency[tuple(i)])
            if seeds[j] <= p_losing_touch:
                if print_flag: print("  ", "Remove edge", tuple(i), "from network")
                model.adjacency[tuple(i)] = 0
            j += 1

        #5. every agent can make one new connection
        indices2 = numpy.argwhere(model.adjacency == 0)
        seeds = numpy.random.random(size=len(indices2))
        j = 0
        for i in indices2:
            # print(i, model.adjacency[tuple(i)])
            if i[[0]] != i[[1]] and seeds[j] <= p_new_connection*cost:
                if print_flag: print("  ", "Add new connection", tuple(i), "to network")
                model.adjacency[tuple(i)] = 1
            j += 1
        return model

    def remove_unconnected(self,model,print_flag=False):
        for ia in reversed(range(model.numagents)):  # remove unconnected agents from the network
            print(ia)
            if numpy.sum(model.adjacency[ia]) == 0 and numpy.sum(model.adjacency[0:model.numagents,ia]) == 0:
                if print_flag: print("  ", "Remove unconnected agent", model.agent[ia].agentID)
                del model.agent[ia]
                model.numagents -= 1
                model.adjacency = numpy.delete(model.adjacency, ia, 0)
                model.adjacency = numpy.delete(model.adjacency, ia, 1)
        return model

    def reconstruct_adj(self,a,print_flag=False):
        adj = numpy.nonzero(a)
        if print_flag: print(numpy.transpose(adj))
        return numpy.transpose(adj)

    def agent_growth(self,model0,cost,p_death,p_mutate,size_input,print_flag=False):
        return model0

    def change_connectivity(self, model, p_add,p_remove, print_flag=False):

        size_input = model.in_bits
        l = [ia for ia in range(size_input,model.numagents)] #do not include input-input couplings
        edge_list = []

        while l != []: #add random connection
            seed = random.random()
            source = random.choice(l)
            l.remove(source)
            neighbors = model.g.get_out_neighbors(source)
            targets = list( set(l) - set(neighbors))
            if targets == []: break
            else:
                weight = model.g.get_out_degrees(targets)
                n = sum(weight)
                if n > 0 and n < 1: weight = weight/n
                target = random.choices(targets,weights = weight)[0]
            try: l.remove(target)
            except: None
            if seed < p_add:
                print("add edge:", (source, target))
                edge_list.append((source,target))
        model.g.add_edge_list(edge_list)

        model.g.set_fast_edge_removal(fast=True)
        for source in model.g.get_vertices():
            for target in model.g.get_out_neighbors(source):
                e = model.g.edge(source,target)
                seed = random.random()
                #if source < size_input and target < size_input: break
                if seed < p_remove:
                    print("remove edge:", e)
                    model.g.remove_edge(e)
                    #break
        if print_flag:
            for v in model.g.vertices():
                print(v, model.g.get_out_neighbors(v))
        return model

    """ --- two preliminary experiments """

    def evolutionary_perception(self,model0,max_time_steps,size_input,alpha,n_pop,n_ind,p_mutate,p_lambda):

        ###--- first preliminary experiment: keep network fixed and optimize strategy via GA --- ###
        self.name = "evolutionary perception"
        model0.majority = False

        print("This experiment uses a genetic algorithm to evolve perceptual strategies on a fixed topology")
        fitness = dict.fromkeys(range(n_pop), {})

        for i_pop in range(n_pop):
            percepts = {} #distinct percepts, will be number of assymptotic states of the network
            for i_ind in range(n_ind):
                model = copy.deepcopy(model0) #story a copy of the network in model

                if i_pop == 0: #iff first in population, randomize strategies

                    for ia in range(model.numagents):

                        thresh = numpy.sum(model.adjacency[ia]) +1 #threshold is the connectivity of agent ia + 1
                        mu1 = numpy.random.randint(1, int(thresh)+1)  # int(thresh)
                        mu2 = numpy.random.randint(mu1, int(thresh)+1)  # model.out_bits
                        model.agent[ia].mu1 = mu1
                        model.agent[ia].mu2 = mu2

                elif i_pop > 0 and i_ind <= int(n_ind*p_lambda):
                    fitness[i_pop][i_ind] = sorted_fitness[i_ind] #keep lambda fittest individual

                elif i_pop > 0 and i_ind > int(n_ind*p_lambda):

                    # select from lambda+mu individuals according fitness
                    ij = self.sexual_selection(fitness[i_pop - 1])
                    # print("Agents selected for mating:", individual,fitness[i_pop-1][individual])

                    # determine splicing number
                    splicing = numpy.random.randint(0, model.numagents + 1)
                    for ia in range(model.numagents):
                        if ia <= splicing:
                            individual = ij[0]
                        else:
                            individual = ij[1]
                        #    print(ia,splicing,individual)
                        mu1 = fitness[i_pop - 1][individual][1][ia]
                        mu2 = fitness[i_pop - 1][individual][2][ia]
                        #rule = fitness[i_pop - 1][individual][2][ia]

                        # introduce mutation
                        seed = numpy.random.random()
                        seed2 = numpy.random.random()
                        # print(mu1, mu2)
                        if p_mutate >= numpy.abs(seed):
                            #mu1 = mu1*seed
                            mu1 = max(1, mu1 - 1) #make mu_1 one step smaller (but at least 1)
                            mu2 = max(mu1+1, mu2 - 1) #make mu_2 one step smaller (bounded below by mu_1+1)
                        elif p_mutate >= numpy.abs(seed2):
                            #mu2 = mu2*seed2
                            thresh = numpy.sum(model.adjacency[ia]) + 1
                            mu1 = min(int(thresh), mu1 + 1) #make mu_1 one step larger (bounded by thresh)
                            mu2 = max(mu1+1, mu2 + 1) #make mu_1 one step larger (bounded by mu_1+1)

                        model.agent[ia].mu1 = mu1
                        model.agent[ia].mu2 = mu2

                if i_pop == 0 or i_ind > int(n_ind*p_lambda):
                    #calculate all possible percepts for per individual (for new individuals, i_ind > n_ind*p_lambda)
                    percepts[i_ind] = self.infer_percepts(model, max_time_steps, size_input,print_flag=False, attractor_flag=False)
                p_i0 = 1 / (2 ** size_input)  # every input is equally likely
                if i_pop == 0 or i_ind > int(n_ind * p_lambda):
                    # calculate MI between input and percepts) per individual (for new individuals, i_ind > n_ind*p_lambda)
                    mi = self.percept_MI(percepts[i_ind], alpha, size_input, p_i0, print_flag=False)
                    if mi < 0: mi = 0 #this shouldn't be possible???

                    #rule = [copy.copy(model.agent[ia].rule) for ia in range(model.numagents)]
                    mu1 = [copy.copy(model.agent[ia].mu1) for ia in range(model.numagents)]
                    mu2 = [copy.copy(model.agent[ia].mu2) for ia in range(model.numagents)]
                    fitness[i_pop][i_ind] = (mi, mu1, mu2)
                if i_pop == 0: #for first print
                    print("Population:", i_pop, "Individual:", i_ind, "MI(I;X):", numpy.around(fitness[i_pop][i_ind][0],3))
                    model.print_network(i_pop, i_ind, fitness[i_pop][i_ind][0])


            sorted_fitness = sorted(fitness[i_pop].values(), key=lambda x: x[0], reverse=True)
            if i_pop == 0:  #save strategy for later
                mu_01 = sorted_fitness[0][1]
                mu_02 = sorted_fitness[0][2]
            # calculate variance and average fitness of MI, if variance is smaller than 1% of average, than abort GA
            pop_fitness = []
            for i_ind in range(n_ind):
                pop_fitness.append(fitness[i_pop][i_ind][0])
            (ave,variance) = global_functions.variance(pop_fitness)
            print("Population:", i_pop, numpy.around([ave,numpy.sqrt(variance)],3))
            if numpy.sqrt(variance) < 0.01*ave or i_pop == n_pop - 1: #print out last
                for i_ind in range(n_ind):
                    print("Population:", i_pop, "Individual:", i_ind, "MI(I;X):", numpy.around(fitness[i_pop][i_ind][0], 3))
                    model.print_network(i_pop, i_ind, fitness[i_pop][i_ind][0])
                break

        # write out parameters of the best:
        sorted_d = sorted(fitness[i_pop].values(), key=lambda x: x[0], reverse=True)
        print("Five best strategies:")
        for i in range(5):
            mi = sorted_d[i][0]
            mu1 = sorted_d[i][1]
            mu2 = sorted_d[i][2]
            print("Mutual information:", numpy.around(mi, 3), "perceptual stragies:", numpy.around(mu1,3), numpy.around(mu2,3))

        print("---")
        print("Test various strategies with probabilistic initial condition:")
        # test majority rule
        for ia in range(model0.numagents):
            nbh = numpy.sum(model0.adjacency[ia])
            # print(nbh)
            if nbh % 2 == 0:
                model0.agent[ia].mu1 = max(1,int(nbh / 2) )
            else:
                model0.agent[ia].mu1 = max(1,int(nbh / 2) + 1)
            model0.agent[ia].mu2 = int(nbh)
            #model0.agent[ia].rule = [0 if (2 ** i - 1) < model0.agent[ia].mu1 else 1 for i in range(2**(1 + int(nbh)))]
        mu1 = [model0.agent[ia].mu1 for ia in range(model0.numagents)]
        mu2 = [model0.agent[ia].mu2 for ia in range(model0.numagents)]

        mi = []

        percepts_majority = self.infer_percepts(model0, max_time_steps, size_input,
                                                            print_flag=False, attractor_flag=False)
        p_i0 = 1 / (2 ** size_input)  # every input is equally likely
        mi.append(self.percept_MI(percepts_majority, alpha, size_input, p_i0, print_flag=False))

        (mean,var) = global_functions.variance(mi)
        print("MI(I;X) under majority rule:", numpy.around([mean,var], 3), "perceptual strategies:",mu1, mu2)
        model0.print_network("maj", 0, mean)

        # test against totalistic rule with threshold = degree
        for ia in range(model0.numagents):
            thresh = numpy.sum(model0.adjacency[ia])
            model0.agent[ia].mu1 = int(thresh)
            model0.agent[ia].mu2 = int(numpy.sum(model0.adjacency[ia]))
            #model0.agent[ia].rule = [0 if (2 ** i - 1) < nbh else 1 for i in range(1 + int(nbh))]
        mu1 = [model0.agent[ia].mu1 for ia in range(model.numagents)]
        mu2 = [model0.agent[ia].mu2 for ia in range(model.numagents)]

        mi = []
        percepts_max = self.infer_percepts(model0, max_time_steps, size_input,
                                                    print_flag=False, attractor_flag=False)
        p_i0 = 1 / (2 ** size_input)  # every input is equally likely
        mi.append(self.percept_MI(percepts_max, alpha, size_input, p_i0, print_flag=False))
        (mean, var) = global_functions.variance(mi)
        print("MI(I;X) under thresh = deg rule:", numpy.around([mean,var], 3), "perceptual strategies:", mu1, mu2)
        model0.print_network("mxt", 0, mean)

        # test best initial strategy
        for ia in range(model0.numagents):
            model0.agent[ia].mu1 = mu_01[ia]
            model0.agent[ia].mu2 = mu_02[ia]
        mi = []
        percepts_initial = self.infer_percepts(model0, max_time_steps, size_input,
                                                        print_flag=False, attractor_flag=False)
        p_i0 = 1 / (2 ** size_input)  # every input is equally likely
        mi.append(self.percept_MI(percepts_initial, alpha, size_input, p_i0, print_flag=False))
        (mean, var) = global_functions.variance(mi)
        print("MI(I;X) for best initial strategy:", numpy.around([mean, var], 3), "perceptual strategies:", mu_01, mu_02)

        # test best evolved strategy
        mu1 = sorted_d[0][1]
        mu2 = sorted_d[0][2]
        for ia in range(model0.numagents):
            model0.agent[ia].mu1 = mu1[ia]
            model0.agent[ia].mu2 = mu2[ia]

        mi = []
        percepts_evolved = self.infer_percepts(model0, max_time_steps, size_input,
                                                        print_flag=False, attractor_flag=False)
        p_i0 = 1 / (2 ** size_input)  # every input is equally likely
        mi.append(self.percept_MI(percepts_evolved, alpha, size_input, p_i0, print_flag=False))
        (mean, var) = global_functions.variance(mi)
        print("MI(I;X) for best evolved strategy:", numpy.around([mean,var], 3), "perceptual strategies:", mu1, mu2)
        # ideal case: for every input one output, i.e. 1-1 mapping from I_a to O; with noise on input
        noise = alpha / 100
        ideal_kernel = global_functions.ideal_kernel(size_input, noise)
        p0 = [1 / (2 ** size_input)] * (2 ** size_input)
        print("Ideal case: max[MI(I;X)]:", numpy.around(global_functions.kernel_MI(p0, ideal_kernel, 2 ** size_input)[0], 3),
              ", p(i) = 1/n, noise level:", noise)
        return

    def evolution_of_topology(self,model0,coeff,p_annealing,max_time_steps,size_input,alpha,
                              n_pop,n_ind,p_lambda,p_mutate):

        ###--- second preliminary experiment: evolve graph topology to maximize MI --- ###
        self.name = "evolution of topology"
        model0.majority=True
        print("This experiment uses a genetic algorithm to evolve the topology of a perceptual network with fixed strategy")
        fitness = dict.fromkeys(range(n_pop), {})
        """ ... """
