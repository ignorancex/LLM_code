import sys
import time
import networkx as nx
from itertools import permutations
from collections import defaultdict

from clara.model import SPECIAL_VARS, Const, Var

# find jaccard distance between two labels
def jaccard(list1, list2):
    l1 = defaultdict(int)
    l2 = defaultdict(int)
    keys = list(set(list1).union(set(list2)))
    top = 0
    bot = 0
    
    for l in list1:
        l1[l] += 1
    for l in list2:
        l2[l] += 1

    for k in keys:
        top += min(l1[k], l2[k])
        bot += max(l1[k], l2[k])
    return top / bot


class GraphMatching():
    def __init__(self, fnc1, fnc2, option):
        self.ICF = fnc2
        self.CF = fnc1
        self.ICG = None
        self.CG = None
        self.ICDict = {}
        self.CDict = {}
        self.removedLocs = {}
        self.option = option
        self.result = []
        self.perms = []

    def createGraphs(self):
        self.ICG = self.makeGraph(self.ICF)
        self.CG = self.makeGraph(self.CF)
        self.CDict = self.createLocLabelDict(self.CG)
        self.ICDict = self.createLocLabelDict(self.ICG)

    # creates a dictionary where the key is the location and the label is the value
    def createLocLabelDict(self, G):
        locLabels = {}
        for name, attr in list(G.nodes(data=True)):
            locLabels[name] = attr['label']
        return locLabels

    # connects the graph by adding the edges
    # locs is the list of locations
    # trans is the dictionary of tranistions and locations
    # labels is the label name for each node
    # G is the graph
    def connect(self, locs, trans, G):
        edges = []
        # traverses through the locations to access the transitions
        for l in locs:
            # extracts the transitions for the location
            t = trans[l]
            true = t[True]
            false = t[False]

            # adds the edges to the list
            if true:
                edges += [(l, true, True)]
            else:
                edges += [(l, 0, True)]
            if false:
                edges += [(l, false, False)]
            else:
                edges += [(l, 0, False)]
        # adds the edges to the graph
        G.add_weighted_edges_from(edges)
        return G

    # gets the label value of an expression based on the type of expression
    def getLableValue(self, exp):
        if isinstance(exp, Var):
            e = exp.name.strip()
            if 'ind#' in e or 'iter#' in e or 'cond' in e:
                return [e]
            return []
        if isinstance(exp, Const):
            value = exp.value
            if '"' in value:
                value = value.replace("'", '').strip()
            return [value]
        name = exp.name
        args = exp.args
        a = []
        for arg in args:
            a += self.getLableValue(arg)
        return [name] + a

    # creates a label for a node
    # calls getLabelValue on each expression in the node and appends it to a list
    def makeLabel(self, exp):
        label = []
        for var, e in exp:
            var = str(var)
            if var in SPECIAL_VARS or 'ind#' in var or 'iter#' in var or 'cond' in var:
                label += [str(var)]
            label += self.getLableValue(e)
        return label

    # creates the graph from the function
    def makeGraph(self, fnc):
        # the multidigraph indicates its directed and can have multiple repeated edges
        G = nx.MultiDiGraph()

        # since a transition can go to None, it gives it the location 0 as 0 can never exist
        G.add_node(0, label='None')

        # extracts the locations and dicts for descriptions, transitions and expressions
        desc = fnc.locdescs
        trans = fnc.loctrans
        exprs = fnc.locexprs
        loc = list(desc.keys())
        allLabels = {}

        # creates a node in the graph for every location with a label, line no, exp and desc
        for l in loc:
            lno = desc[l].split(" ")[-1]
            exp = exprs[l]
            label = self.makeLabel(exp)
            label = ','.join(list(label))
            if not label:
                label = 'EMPTY'
            allLabels[l] = label

            G.add_node(l, label=label, desc=desc[l], lno=lno, exprs=exp)

        # calls connect to add the edges to the graph
        G = self.connect(loc, trans, G)
        return G

    #  used to visualise the graph
    def printGraph(self, G):
        for name, attr in list(G.nodes(data=True)):
            print('loc : ', name)

            if len(attr):
                print('label : ', attr['label'])

            edges = list(G.edges(name, data=True))
            for e in edges:
                _, to, trans = e
                print(trans['weight'], ' : ', to)
            print('\n')

    def createMatchDict(self):
        G1 = self.CG
        G2 = self.ICG
        d1 = self.CDict
        d2 = self.ICDict
        labelSim = {0: {0: 1}}

        # checks to see which graph is shorter
        l1 = list(G1.nodes())
        l2 = list(G2.nodes())
        len_l1 = len(l1)
        len_l2 = len(l2)
        self.shorter = 1
        if len_l1 < len_l2:
            self.result += ['DEL', len_l2-len_l1]
        elif len_l1 > len_l2:
            temp = l2
            l2 = l1
            l1 = temp
            self.result += ['ADD', len_l1-len_l2]
            self.shorter = 2

        # removing node 0 from comparisons because that will always match with each other
        l1.remove(0)
        l2.remove(0)
        # create label scores between every node
        for n1 in l1:
            labelSim[n1] = {}
            if self.shorter == 2:
                lab1 = d2[n1]
            else:
                lab1 = d1[n1]
            for n2 in l2:
                if self.shorter == 2:
                    lab2 = d1[n2]
                else:
                    lab2 = d2[n2]
                val = jaccard(lab1.split(','), lab2.split(','))
                labelSim[n1][n2] = val

        possibleMatch = {0: [0]}
        # sort possible matches for each node by label value
        for n1 in labelSim:
            options = [k for k, _ in sorted(
                labelSim[n1].items(), key=lambda item: item[1], reverse=True)]
            possibleMatch[n1] = options
        
        self.permBackTrack([0] + l1, possibleMatch, {}, labelSim)
        
        (bestMatch, score) = self.findBestMatch(self.perms, labelSim)
        score /= len(possibleMatch)

        print("Score:", score)
        if score < 0.6:
            print('SCORE TOO LESS')
            sys.exit(0)
        self.createModel(bestMatch)

    def findBestMatch(self, phi, labelScores):
        G1 = self.CG
        G2 = self.ICG
        bestMatch = []
        score = 0
        
        # traversing a possible match for the graphs
        for match in phi:
            currScore = 0
            # traversing each node match
            for n1 in match:
                if n1 == 0:
                    currScore += 1
                    continue
                n2 = match[n1]
                labelDist = labelScores[n1][n2]
                if self.option == 1:
                    edges1 = []
                    edges2 = []
                    if self.shorter == 2:
                        edges1 = list(G1.edges(n2, data=True))
                        edges2 = list(G2.edges(n1, data=True))
                    else:
                        edges1 = list(G1.edges(n1, data=True))
                        edges2 = list(G2.edges(n2, data=True))

                    _, t1, _ = edges1[0]
                    _, f1, _ = edges1[1]
                    _, t2, _ = edges2[0]
                    _, f2, _ = edges2[1]

                    edgeDist = 0
                    if self.shorter == 2:
                        if (t1 == match[t2]):
                            edgeDist += 1
                        if (f1 == match[f2]):
                            edgeDist += 1
                    else:
                        if (t2 == match[t1]):
                            edgeDist += 1
                        if (f2 == match[f1]):
                            edgeDist += 1

                    edgeDist *= 0.5
                    currScore += (labelDist + edgeDist) / 2
                else:
                    currScore += labelDist
            # if currScore is greater than this is the best match of graphs
            if currScore > score:
                score = currScore
                bestMatch = match
        return (bestMatch, score)

    def permBackTrack(self, U, possibleMatch, phi, labelSim):
        if len(U) == len(phi):
            if phi not in self.perms:
                self.perms += [dict(phi)]
            if len(self.perms) == 1000:
                return True
            return False
        unmappedU = set(U) - set(phi.keys())

        for currU in unmappedU:
            for currV in possibleMatch[currU]:
                if currV in phi.values():
                    continue
                temp_phi = phi
                temp_phi[currU] = currV
                flag = self.permBackTrack(U, possibleMatch, temp_phi, labelSim)
                if flag:
                    return True
        return False

 # based on the correct graph and the matching, it recreates the incorrect model
    # if the correct graph is longer, nodes are added to the incorrect graph
    # if the correct graph is shorter, nodes are removed from the incorrect graph
    def createModel(self, bestMatch):
        G1 = self.CG
        G2 = self.ICG
        # intialize the following dicts for new expressions, descriptions and transitions of the model
        new_e2 = {}
        new_d2 = {}
        new_t2 = {}
        if self.shorter == 1:
            initloc = bestMatch[self.CF.initloc]
        else:
            # find the maximum location number so we don't have repeated locations
            maxLoc = max(G2.nodes())
            lastLoc = maxLoc
            mappedG1Locs = set(bestMatch.values())
            unmappedG1Locs = list(set(G1.nodes()) - mappedG1Locs)
            new_locs = {}

            # create a mapping of new locations to be added from G1 -> G2
            for l in unmappedG1Locs:
                maxLoc += 1
                new_locs[l] = maxLoc
            
            new_locs.update({v:k for k, v in bestMatch.items()})

        for loc1 in bestMatch:
            if loc1 == 0:
                continue
            loc2 = bestMatch[loc1]

            # if G2 is smaller, reverse the locations as bestMatch is a map from G2 -> G1
            if (self.shorter == 2):
                temp = loc1
                loc1 = loc2
                loc2 = temp
                if loc1 == self.CF.initloc:
                    initloc = loc2
            new_e2[loc2] = G2.nodes[loc2]['exprs']
            new_d2[loc2] = G2.nodes[loc2]['desc']

            edges = list(G1.edges(loc1, data=True))
            _, t_edge, _ = edges[0]
            _, f_edge, _ = edges[1]
            if self.shorter == 1:
                t = None if bestMatch[t_edge] == 0 else bestMatch[t_edge]
                f = None if bestMatch[f_edge] == 0 else bestMatch[f_edge]
            else:
                t = None if new_locs[t_edge] == 0 else new_locs[t_edge]
                f = None if new_locs[f_edge] == 0 else new_locs[f_edge]
            new_t2[loc2] = {True: t, False: f}

        # add the remaining locations since G2 is smaller
        if self.shorter == 2:
            for currLoc in unmappedG1Locs:
                edges = list(G1.edges(currLoc, data=True))
                _, finalTrue, _ = edges[0]
                _, finalFalse, _ = edges[1]
                trans = finalTrue
                mainLoc = currLoc
                desc = 'At the end of the function'
                flag = False
                while(trans == 0 or trans not in mappedG1Locs):
                    if trans == lastLoc:
                        flag = True
                        break
                    if trans < currLoc or trans == 0:
                        _, false_edge, _ = edges[1]
                        if trans == 0 and false_edge == 0:
                            flag = True
                            break
                        currLoc = trans
                        trans = false_edge
                    else:
                        currLoc = trans
                        edges = list(G1.edges(trans, data=True))
                        _, true_edge, _ = edges[0]
                        trans = true_edge
                newG2Loc = new_locs[mainLoc]
                new_e2[newG2Loc] = []
                new_t2[newG2Loc] = {True: None if finalTrue == 0 else new_locs[finalTrue], False: None if finalFalse == 0 else new_locs[finalFalse]}
                
                if not flag:
                    desc = G2.nodes[new_locs[trans]]['desc']
                new_d2[newG2Loc] = desc
        if self.shorter == 1:
            # adds all the expressions into a list so they can be adde to the feedback
            remLocs = list(self.ICF.locexprs.keys() - bestMatch.values())
            for l in remLocs:
                self.removedLocs[l] = (self.ICF.locexprs[l], self.ICF.locdescs[l])
        self.ICF.initloc = initloc
        self.ICF.locexprs = new_e2
        self.ICF.loctrans = new_t2
        self.ICF.locdescs = new_d2

        print('\n', bestMatch)
