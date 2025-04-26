#!/usr/bin/env python
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
"""

import time, sys, getopt, argparse 
import Environment_rnd, selection_rnd
import tables
from utilities import bps2string

def main(argv):

    
    parser = argparse.ArgumentParser(description='Ensemble_Q')
    parser.add_argument('-i',        help = 'Location of the input files', required = True)
    parser.add_argument('-o',        help = 'Location of the output files', required = True)
    parser.add_argument('-np',       help = 'Number of base predictors', required = True)
    parser.add_argument('-fold',     help = 'Fold', required = True)
    parser.add_argument('-m',        help = 'Metric used for performance', required = True)
    parser.add_argument('-age',      help = 'Pretraining (moves) prior to convergence of the q-table', required = True, default = 0)  
    parser.add_argument('-seed',     help = 'Seed for randomization', required = True)
    parser.add_argument('-epsilon',  help = 'Epsilon (exploration) probability for QL', required = False, default = 0.1)    
    parser.add_argument('-rule',     help = 'Combination rule for ensembles', required = True) 
    parser.add_argument('-strategy', help = 'Strategy', required = True)
    parser.add_argument('-start',    help = 'Start: 0 (zero) or best (most accurate base predictor)', required = True) 
    parser.add_argument('-exit',     help = 'Exit: 0 or 1', required = True) 
    parser.add_argument('-algo',     help = 'Algorithm: Q (QL) or S (SARSA)', required = True) #SARSA not available in this release
    parser.add_argument('-conv',     help = 'Consecutive iterations of picking the same ensemble', required = True, default = 1000)
    #parser.add_argument('-run',     help='Independent runs', required=False)
    args = parser.parse_args()
 


    """
    print ("    Fold: %s" % args.fold)
    print ("    # bp: %s" % args.np)
    print ("    Perf: %s" % args.m)
    print ("   INPUT: %s" % args.o)
    print ("  OUTPUT: %s" % args.o)
    print ("    Seed: %s" % args.seed)
    print ("     Age: %s" % args.age)
    print (" Epsilon: %s" % args.epsilon)    
    print ("    Rule: %s" % args.rule)
    print ("Strategy: %s" % args.strategy)
    print ("   Start: %s" % args.start)
    print ("    Exit: %s" % args.exit)
    print ("    Algo: %s" % args.algo)
    print ("    Conv: %s" % args.conv)
    """


    fold 	= int(args.fold)
    num_pred 	= int(args.np)
    metric 	= args.m
    in_dir 	= args.i
    out 	= args.o
    age 	= int(args.age)
    seed 	= int(args.seed)
    epsilon 	= float(args.epsilon)    
    rule 	= args.rule
    strategy 	= args.strategy
    start 	= args.start
    exit 	= float(args.exit)
    conv 	= int(args.conv)
    #run         = int(args.run)
    algo 	= args.algo
    assert strategy in ['pessimyule', 'pessimkappa', 'pessimcorr', 'pessimcos', 'pessimeuclid', 'gsernd', 'greedy', 'pessimistic', 'backtrack', 'diversity', 'correlation', 'euclidean', 'yule', 'diversityrnd', 'euclideanrnd', 'correlationrnd', 'greedyrnd', 'pessimisticrnd', 'backtrackrnd', 'yulernd', 'kapparnd']
    assert algo     in ['Q']
    assert start    in ['0', 'best']
    assert exit     in [0, 1]
    assert rule     in ['WA', 'L2']


    filename = '%s/bp%s_fold%i_seed%i_epsilon%.2f_pre%i_conv%i_exit%i_%s_%s_%s_start-%s.%s' % (out, num_pred, fold, seed, epsilon, age, conv, exit, strategy, rule, algo, start, metric)
    world = Environment_rnd.World(num_pred, start, exit, in_dir, seed, fold, rule, metric)
    start_time = time.time()
    #instruction = ("agent = selection_rnd.Agent(epsilon, seed, strategy, age, conv)")
    #exec("print(\"agent = selection_rnd.Agent(epsilon, seed, strategy, age, conv)\")")
    #exec(instruction) 
    agent = selection_rnd.Agent(epsilon, seed, strategy, age, conv)
    agent.setWorld(world)
    #print(agent.world.printClassifiers())
    agent.converge(conv)
    ensemble = agent.getEns()
 
    seconds  = time.time() - start_time
    result   = 'Fold_%i age=%s num_episodes=%i (visited=%f%% (%i nodes), explored=%f%% (%i nodes)) (val=%f) (test=na) (bestBP %s=na) (FE=na) :: %r [%s]\n%s'  % (fold, agent.age, agent.num_episodes, float(len(agent.getVisitedStates()))/float(2**num_pred)*100, len(agent.getVisitedStates()), float(len(agent.getExploredStates()))/float(2**num_pred)*100, len(agent.getExploredStates()), agent.getEnsPerf(), agent.world.getBestBP(), ensemble, time.strftime('%H:%M:%S', time.gmtime(seconds)), bps2string(list(agent.world.predictors)))

    with open(filename, "w+") as f:
        f.write(result)
    f.close()
    print("\t%s (%s)"  % (filename, (time.strftime('%H:%M:%S', time.gmtime(seconds)))))

if __name__ == "__main__":
   main(sys.argv[1:])



