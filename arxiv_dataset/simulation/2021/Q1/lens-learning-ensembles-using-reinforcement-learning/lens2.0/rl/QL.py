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
    Inspiration: https://github.com/studywolf/blog/tree/master/RL
"""
import random
import collections
from collections import OrderedDict


class QL:
    def __init__(self, epsilon, seed, alpha=0.1, gamma=0.9):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        self.exploration = False
        #print "\n\ne=%f, seed=%s, alpha=%f, gamma=%f" % (epsilon, seed, alpha, gamma)

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)
        #return self.q.get((state, action), 1.0)

    def update(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            #print "utility(%r,%r) = %f" %(state, action, reward)
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, actions):
        #random.seed(self.seed) #for reproducibility  
        if (random.random() < self.epsilon): #exploration
            action = random.choice(actions) 
            self.exploration = True
        else: #exploitation
            self.exploration = False
            if (self.q == {}):
                action = random.choice(actions)
            else:
                q = [self.getQ(state, a) for a in actions]
                maxQ = max(q)
                count = q.count(maxQ)   
                if count > 1:
                    ax = []
                    for a in actions:
                        if self.getQ(state, a) == maxQ:
                            ax.append((state,a))
                    action = random.choice(ax)[1]
                else:
                    for a in actions:
                        if self.getQ(state, a) == maxQ:
                            action = a
        return action

  
    #QL
    def learnQL(self, state1, action1, reward, state2, actions):
        maxqnew = max([self.getQ(state2, a) for a in actions])
        self.update(state1, action1, reward, reward + self.gamma*maxqnew)

    def printQ(self):
        od = collections.OrderedDict(sorted(self.q.items()))
        for (key, val) in od.items():
            print("%r:%r" % (key, val))








