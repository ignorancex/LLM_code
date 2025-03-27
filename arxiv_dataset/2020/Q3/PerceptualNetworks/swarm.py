import numpy
import copy
import sys
import itertools
import networkx
import global_functions
import random
from datetime import datetime

class environment:

    # introduce an "environment" for providing feedback
    def __init__(self,input_size,print_flag=False):

        self.function={}
        self.input_size = input_size
        # initialize rule:
        for key in itertools.product({0, 1}, repeat = self.input_size):
            self.function[key] = list(numpy.random.randint(0, 2, 3))
        if print_flag:
            print("Initialize environment with the following rule:")
            for key in self.function:
                print(key, self.function[key])

    def feedback(self,input,delay=False,print_flag=False):
        output = list(input)
        if input in self.function and not delay:
            output = self.function[input]
        elif input not in self.function and not delay:
            print("Error: input unknown")
        if print_flag: print("Environmental feedback:", input, "->", output)
        return output

class agent:

    def __init__(self,ia,node,bits,nbh,in_node,mu1,mu2,majority=False):
        self.agentID = ia
        self.node = node
        self.state = 0
        self.x = 0
        self.bits = bits
        self.nbh = int(nbh)
        self.in_node = in_node
        self.nt = 0
        self.mu1 = mu1
        self.mu2 = mu2
        self.fitness = 0
        self.majority = majority
        #self.mu3 = mu3
        #self.out_nodes = out_nodes
        self.rule = [numpy.random.randint(0,2) for i in range(2**self.nbh+1)]  #create a nbh sized binary number

    def update(self,input,adj=0,print_flag=False):
        x0 = self.x
        k = input
        if x0 == 0 and k >= self.mu1 and k <= self.mu2: self.x = 1 #only update if node was deactivitated in last step
        else: self.x = 0
        if print_flag: print("state update:", x0, "->", self.x)
        return self.x

class network:

    def __init__(self,numagents,bits,in_nodes,print_flag=False):

        """ initialize empty list of perceptual agents (nodes)"""
        self.numagents = numagents
        self.agent = []
        self.in_bits = in_nodes

        """ initialize empty dict for trajectories"""
        self.trajectories={}
        self.accepted_changes = 0
        self.learning_rate = numpy.zeros((3,1),dtype=int)

        """ initialize network topology """
        self.adjacency = numpy.zeros((numagents,numagents),dtype=int)


        # Erdös Renyi network; symmetric G(n,p) with p=0.5;
        # if p > ln(n)/n graph is almost surely connected
        
        p0=0.75
        for ia1 in range(self.numagents):
            for ia2 in range(ia1+1,self.numagents):
                if ia1 < 3 and ia2 < 3: self.adjacency[ia1,ia2] = 1 #fully connected in_nodes
                else:
                    pER = numpy.random.random()
                    if pER <= p0: self.adjacency[ia1,ia2] = 1
                self.adjacency[ia2,ia1] = self.adjacency[ia1,ia2] #symmetrize



        """#complete graph (dense)
        for ia1 in range(self.numagents):
            for ia2 in range(ia1+1,self.numagents):
                self.adjacency[ia1,ia2] = 1
                self.adjacency[ia2,ia1] = 1
        """


        """
        #4-lattice
        for ia in range(self.numagents):
            if ia > 1 and ia < self.numagents-2:
                ia1 = ia + 1
                ia2 = ia + 2
                ia3 = ia - 1
                ia4 = ia - 2

            elif ia == 0:
                ia1 = ia + 1
                ia2 = ia + 2
                ia3 = self.numagents-1
                ia4 = self.numagents-2

            elif ia == 1:
                ia1 = ia + 1
                ia2 = ia + 2
                ia3 = ia - 1
                ia4 = self.numagents-1

            elif ia == self.numagents-2:
                ia1 = ia + 1
                ia2 = 0
                ia3 = ia - 1
                ia4 = ia - 2
            elif ia == self.numagents-1:
                ia1 = 0
                ia2 = 1
                ia3 = ia- 1
                ia4 = ia -2

            self.adjacency[ia,ia1] = 1
            self.adjacency[ia1,ia] = 1
            self.adjacency[ia,ia2] = 1
            self.adjacency[ia2,ia] = 1
            self.adjacency[ia,ia3] = 1
            self.adjacency[ia3,ia] = 1
            self.adjacency[ia, ia4] = 1
            self.adjacency[ia4, ia] = 1
        """

        
        #SF-network, BA(m=1)
        """
        m=1
        main_graph = networkx.barabasi_albert_graph(self.numagents, m)
        self.adjacency = networkx.to_numpy_array(main_graph)
        """

        # density is number of edges/number of possible edges
        self.graph_density = numpy.sum(self.adjacency)/(self.numagents*(self.numagents-1) )


        """ initialize agents: """
        iagents = range(self.in_bits)
        majority = True
        for ia in range(self.numagents):
            nbh = 3 #ALT: take mean neighborhood from graph; nbh = global_functions.dict_from_adj(self.adjacency)
            mu1 = numpy.random.randint(1, nbh + 1)  # int(thresh)
            mu2 = numpy.random.randint(mu1, nbh + 1)  #
            if ia in iagents: i = True
            else: i = False
            self.agent.append(agent(ia,ia,bits,nbh,i,mu1,mu2,majority=majority)) #initialize with majority rule
            #print(ia,self.agent[ia].rule)

        if print_flag:
            adj_dict = global_functions.dict_from_adj(self.adjacency)
            numagents = 0
            for ia in range(len(self.adjacency)):
                if numpy.sum(self.adjacency[ia]) != 0 or numpy.sum(self.adjacency[:, ia]) != 0: numagents += 1
            print("Initialize network with", self.numagents, "agents on a graph with"
                  , len(adj_dict), "edges and density of", numpy.around(self.graph_density,2))
            print("Input agents:", iagents)
            print("Average degree:", numpy.around(numpy.sum(self.adjacency)/self.numagents,2))

            if self.numagents <= 12:
                for l in self.adjacency: print(l)
            else:
                for key in adj_dict: print(key,"->",adj_dict[key])

            print("Perceptual rule of type:")
            if majority: print("majority rule: x -> 1: if sum(input) >= out_degree/2")
            else:
                print("x -> 1: if mu1 <= sum(input) <= mu2")
                print("mu1 =",[self.agent[ia].mu1 for ia in range(self.numagents)],
                      "mu2 =",[self.agent[ia].mu2 for ia in range(self.numagents)])
            #for key in itertools.product({0, 1}, repeat = self.out_bits):
            #    print(key, self.agent[0].update(key))
            print("Network successfully initialized")

    """ --- various network functions --- """

    def create_gene(self, bits, prob_kernels):
        gene = [0] * (2 ** bits)  # gene that controls the strategy
        if prob_kernels == False:  # exactly one state is chosen (i.e., kernel = mapping)
            j = numpy.random.randint(0, 2 ** bits)
            gene[j] = 1
        else:
            v = numpy.random.random(size=2 ** bits).tolist()
            gene = global_functions.normalize(v)  # kernel is a normalized sum
        return gene

    def add_agent(self,parent,p_mutate,print_flag=False):
        ia = self.numagents
        bits = self.agent[parent].bits
        mu1 = self.agent[parent].mu1
        mu2 = self.agent[parent].mu2
        #random mutation
        seed1 = numpy.random.random()
        seed2 = numpy.random.random()
        if p_mutate >= seed1:
            mu1 = max(1, mu1 - 1)
            mu2 = max(mu1+1, mu2 - 1)
        if p_mutate >= seed2:
            mu2 = mu2 + 1
            mu2 = max(mu1 + 1, mu2 + 1)
        ###
        self.agent.append(agent(ia,ia,bits,nbh=1,in_node=False,mu1=mu1,mu2=mu2))
        self.numagents += 1
        self.agent[ia].x = 0

    def remove_agent(self,a1,fast=True,print_flag=False):
        last = self.numagents - 1
        if fast: self.agent[a1] = copy.deepcopy(self.agent[last]) #fast edge removal: exchange a1 and last agent
        del self.agent[last]

        self.numagents = len(self.agent)

        if print_flag:
            print("Update network:", self.numagents)

    def print_network(self,n_pop,n_ind,mi=0.):

        # print out to a file
        numagents = copy.copy(self.numagents)
        dt = datetime.now()

        fileName = "perceptual network/states/" + dt.strftime("%Y_%m%d_%H%M") + ".csv"
        outFile = open(fileName, "a")
        if n_pop == 0 and n_ind == 0:
            #outFile = open(fileName, "w")
            outFile.write("***, " + fileName + "\n")  # header
            #outFile.write("***, numagents" + "\n")  # header
            #outFile.write(str(numagents) + "\n")  # header

            #write out adjacency
            outFile.write("***, numagents, n_pop, n_ind, graph_density, adjacency" + "\n")
            outFile.write(str(numagents) + ", " + str(n_pop) + ", " + str(n_ind) + ", " + str(self.graph_density) + "\n")  # header
            for line in self.adjacency: outFile.write(str(line) + "\n")

            #writes out the parameters of the network, average thresholds and relative entropy
            outFile.write("***, thresholds, averages and mutual information (per population and individual)" + "\n")

        #else:
        mu_1 = [self.agent[ia].mu1 for ia in range(self.numagents)]
        mu_2 = [self.agent[ia].mu2 for ia in range(self.numagents)]
        ave_mu1 = global_functions.variance(mu_1)
        ave_mu2 = global_functions.variance(mu_2)
        outFile.write(str(n_pop) + ", " + str(n_ind) + ", " + str(mu_1)[1:-1] + ", " + str(mu_2)[1:-1] + ", "
                      + str(numpy.around(ave_mu1,2))[1:-1] + ", " + str(numpy.around(ave_mu2,2))[1:-1] + ", "
                      + str(numpy.around(mi,3)) + "\n")



