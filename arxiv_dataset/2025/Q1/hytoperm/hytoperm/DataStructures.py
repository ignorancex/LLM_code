'''
The tree data structures used for the high level planner. 
'''

# external imports
from __future__ import annotations
import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import Set, List

# internal imports
from .World import Region
from .PyPlotHelpers.Plotters import PlotObject, getAxes


class Node:
    def __init__(
            self, 
            p : np.ndarray, 
            r : Set[Region], 
            active_region_to_parent : Region = None, 
            costToRoot : float = 0, 
            costToParent : float = 0
            ) -> None:
        self._r : Set[Region] = set()                                           # regions
        self._ctr : float = costToRoot                                          # cost to root
        self._ctp : float = costToParent                                        # cost to parent
        self._rtp : Region = None                                               # active region to parent cost
        self._p : np.ndarray = None
        self.updatePosition(p)         
        self.updateRegions(r, active_region_to_parent)         

    # getters
    def p(self) -> np.ndarray:
        return self._p
    
    def regions(self) -> Set[Region]:
        return self._r
        
    def costToRoot(self) -> float:
        return self._ctr
    
    def costToParent(self) -> float:
        return self._ctp

    def activeRegionToParent(self) -> Region:
        return self._rtp
    
    # modifiers
    def activate_region_to_parent(self, r : Region) -> None:
        if not isinstance(r, Region):
            raise ValueError("Region must be a Region object.")
        if not r in self._r:
            raise ValueError("Region must be affiliated with node.")
        self._rtp = r

    def updatePosition(self, p : np.ndarray) -> None:
        if not isinstance(p, np.ndarray):
            raise ValueError("Position must be a numpy array.")
        self._p = p

    def updateRegions(self, r : Set[Region], rtp : Region = None) -> None:
        if not isinstance(r, set):
            try:
                r = set(r)
            except:
                raise ValueError("Regions must be a set of Region objects.")
        for region in r:
            self.addRegion(region)
        if rtp is not None:
            self.activate_region_to_parent(rtp)

    def addRegion(self, r : Region) -> None:
        if not isinstance(r, Region):
            raise ValueError("Region must be a Region object.")
        self._r.add(r)
        
    def updateCostToRoot(self, costToRoot : float):
        self._ctr = costToRoot

    def updateCostToParent(self, costToParent : float):
        self._ctp = costToParent

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self.p()[0], self.p()[1], **kwargs))

    def plotAffiliatedRegions(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for r in self.regions():
            po.add(ax.plot(r.p()[0], r.p()[1], 'gx'))
        return po
'''
pyTree: A list-derived TREE data structure in Python 

Created on Aug 21, 2012

@author: yoyzhou

Modifications made (Jonas Hall):
    - every data property is a Node object
    - new methods:
        - setParent
        - hasParent
        - getData
        - getNodesInRegion
        - getNeighborhood
        - cut
        - getPathToRoot
        - plot functions
'''
class Tree(object):
    '''
        A Python implementation of Tree data structure 
    '''
     
    def __init__(self, data : Node, children = None):
        '''
        @param data: content of this node
        @param children: sub node(s) of Tree, could be None, child (single) or 
            children (multiple)
        '''
        self.__data = data
        self.__children : List[Tree] = []                                       # private children attribute
        self.__parent : Tree = None                                             # private parent attribute
                
        if children: #construct a Tree with child or children
            if isinstance(children, Tree):
                self.__children.append(children)
                children.__parent = self 
                
            elif isinstance(children, collections.Iterable):
                for child in children:
                    if isinstance(child, Tree):
                        self.__children.append(child)
                        child.__parent = self
                    else:
                        raise TypeError('Child of Tree should be a Tree type.')      
            else:
                raise TypeError('Child of Tree should be a Tree type')
    
    def __setattr__(self, name, value):
        
            
        """
            Hide the __parent and __children attribute from using dot assignment
            To add __children, please use addChild or addChildren method
            And node's parent isn't assignable
        """
            
        if name in ('parent', '__parent', 'children'):
            raise AttributeError(
                "To add children, use addChild or addChildren method."
                )
        elif name in ('__data', 'data'):
            if not isinstance(value, Node):
                raise TypeError("Data must be a Node object.")
            self.__data = value
        else:
            super().__setattr__(name, value)
            
    def __str__(self, *args, **kwargs):
        
        return self.data.__str__(*args, **kwargs)

    def addChild(self, child):
        """
            Add one single child node to current node
        """
        if isinstance(child, Tree):
                self.__children.append(child)
                child.__parent = self
        else:
                raise TypeError('Child of Tree should be a Tree type')
            
    def addChildren(self, children):
        """
            Add multiple child nodes to current node
        """
        if isinstance(children, list):
                for child in children:
                    if isinstance(child, Tree):
                        self.__children.append(child)
                        child.__parent = self
                    else:
                        raise TypeError('Child of Tree should be a Tree type.')      

    def setParent(self, parent, costToParent : float):
        """
            Set node's parent node.

            Parameters:
            - parent (Tree): The parent node to be set.
            - costToParent (float): The time to reach parent node.
        """
        if not isinstance(parent, Tree):
            raise TypeError('Parent of Tree should be a Tree type.')
            
        parent.addChild(self)
        self.getData().updateCostToParent(costToParent)
        costToRoot = costToParent + parent.getData().costToRoot()
        self.getData().updateCostToRoot(costToRoot)

    def getParent(self) -> Tree:
        """
            Get node's parent node.
        """
        return self.__parent
    
    def getChild(self, index) -> Tree:
        """  
            Get node's No. index child node.
            @param index: Which child node to get in children list, starts with 
                0 to number of children - 1
            @return:  A Tree node presenting the number index child
            @raise IndexError: if the index is out of range 
        """
        try:
            return self.__children[index]
        except IndexError:
            raise IndexError("Index starts with 0 to number of children - 1")
        
    def getChildren(self) -> List[Tree]:
        """
            Get node's all child nodes.
        """
        return self.__children

    def hasParent(self) -> bool:
        """
            Determine whether node has a parent node or not.
        """
        if self.__parent is None:
            return False
        else:
            return True

    def getData(self) -> Node:
        """
            Get node's data.
        """
        return self.__data
    
    def getNode(self, content, includeself = True) -> Node:
        """
            Get the first matching item(including self) whose data is equal to 
            content. Method uses data == content to determine whether a node's 
            data equals to content, note if your node's data is self defined 
            class, overriding object's __eq__ might be required. Implement Tree 
            travel (level first) algorithm using queue

            @param content: node's content to be searched 
            @return: Return node which contains the same data as parameter 
                content, return None if no such node
        """
        
        nodesQ = []
        
        if includeself:
            nodesQ.append(self)
        else:
            nodesQ.extend(self.getChildren())
            
        while nodesQ:
            child = nodesQ[0]
            if child.data == content:
                return child
            else:
                nodesQ.extend(child.getChildren())
                del nodesQ[0]

    def getNodesInRegion(self, r : Region) -> List[Tree]:
        """
            Get all nodes in a region.
        """
        nodes : List[Tree] = []

        sNodes = self.getRoot().getChildren()
        sNodes.append(self.getRoot())
        
        for n in sNodes:
            if r in n.getData().regions():
                nodes.append(n)
        
        return nodes

    def getNeighborhood(self, n : Node) -> dict:
        """
            Get all nodes in a region.
        """
        neighborhood = {}
        queue = [self.getRoot()]

        while queue:
            node = queue.pop(0)
            for rq in node.getData().regions():
                for rn in n.regions():
                    if rq == rn:
                        if node in neighborhood:
                            ValueError(
                                "Multiple regions of intersection... " +
                                " this should not occur."
                                )
                        neighborhood[node] = rq
            queue.extend(node.getChildren())
        return neighborhood

    def cut(self, c : float) -> None:
        """
            Cuts all branches of higher cost than c.
        """
        if self.getData().costToRoot() > c:
            self_child = self.getParent().getChildren().index(self)
            self.getParent().delChild(self_child)
        else:
            for child in self.getChildren():
                child.cut(c)

    def delChild(self, index) -> None:
        """  
            Delete node's No. index child node.
            @param index: Which child node to delete in children list, starts 
                with 0 to number of children - 1
            @raise IndexError: if the index is out of range 
        """
        try:
            del self.__children[index]
        except IndexError:
            raise IndexError("Index starts with 0 to number of children - 1")
    
    def delNode(self, content) -> None:
        """
            Delete the first matching item(including self) whose data is equal 
            to content. Method uses data == content to determine whether a 
            node's data equals to content, note if your node's data is self 
            defined class, overriding object's __eq__ might be required. 
            Implement Tree travel (level first) algorithm using queue

            @param content: node's content to be searched 
        """
        
        nodesQ = [self]
        
        while nodesQ:
            child = nodesQ[0]
            if child.data == content:
                if child.isRoot():
                    del self
                    return
                else:
                    parent = child.getParent()
                    parent.delChild(parent.getChildren().index(child))
                    return
            else:
                nodesQ.extend(child.getChildren())
                del nodesQ[0]
                
    def getRoot(self):
        """
            Get root of the current node.
        """
        if self.isRoot():
            return self
        else:
            return self.getParent().getRoot()
                
    def isRoot(self):
        """
            Determine whether node is a root node or not.
        """
        if self.__parent is None:
            return True
        else:
            return False
        
    def hasChild(self) -> bool:
        """
            Determine whether node has children nodes or not.
        """
        if len(self.__children) == 0:
            return False
        else:
            return True
    
    def isBranch(self):
        """
            Determine whether node is a branch node or not.
        """
        if self.__children == []:
            return True
        else:
            return False
        
    def prettyTree(self):
        """"
            Another implementation of printing tree using Stack
            Print tree structure in hierarchy style.
            For example:
                Root
                |___ C01
                |     |___ C11
                |          |___ C111
                |          |___ C112
                |___ C02
                |___ C03
                |     |___ C31
            A more elegant way to achieve this function using Stack structure, 
            for constructing the Nodes Stack push and pop nodes with additional 
            level info. 
        """

        level = 0        
        NodesS = [self, level]   #init Nodes Stack
        
        while NodesS:
            # head pointer points to the first item of stack
            # can be a level identifier or tree node 
            head = NodesS.pop() 
            if isinstance(head, int):
                level = head
            else:
                self.__printLabel__(head, NodesS, level)
                children = head.getChildren()
                children.reverse()
                
                # push level info if stack is not empty
                if NodesS:
                    NodesS.append(level)    
                
                # add children if has children nodes 
                if children:          
                    NodesS.extend(children)
                    level += 1
                    NodesS.append(level)
    
    def nestedTree(self):
        """"
            Print tree structure in nested-list style.
            For example:
            [0] nested-list style
                [Root[C01[C11[C111,C112]],C02,C03[C31]]]
            """
        
        NestedT = ''  
        delimiter_o = '['
        delimiter_c = ']'                                                                                  
        NodesS = [delimiter_c, self, delimiter_o]
                                                                                            
        while NodesS:
            head = NodesS.pop()
            if isinstance(head, str):
                NestedT += head
            else:
                NestedT += str(head.data)
                
                children = head.getChildren()
            
                if children:          #add children if has children nodes 
                    NodesS.append(delimiter_c)
                    for child in children: 
                        NodesS.append(child)
                        NodesS.append(',')
                    NodesS.pop()
                    NodesS.append(delimiter_o) 
               
        print(NestedT)
          
    def __printLabel__(self, head, NodesS, level):
        """
           Print each node
        """
        leading = '' 
        lasting = '|___ '
        label = str(head.data)
        
        if level == 0:
            print(str(head))
        else:
            for l in range(0, level - 1):
                sibling = False
                parentT = head.__getParent__(level - l)
                for c in parentT.getChildren():
                    if c in NodesS:
                        sibling = True
                        break
                if sibling:
                    leading += '|     '
                else:
                    leading += '     '
            
            if label.strip() != '': 
                print('{0}{1}{2}'.format( leading, lasting, label))
        
    def __getParent__(self, up):
        parent = self;
        while up:
            parent = parent.getParent()
            up -= 1
        return parent

    def plotPathToParent(
            self, 
            ax : plt.Axes = None, 
            annotate_cost = False, 
            plot_direction = False,
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        """
            Plots the path to the parent.
        """
        if self.getParent() is None:
            return
        
        p = self.getData().p()
        q = self.getParent().getData().p()
        
        po = PlotObject(ax.plot([p[0], q[0]] , [p[1], q[1]], **kwargs))
        if annotate_cost:
            normal = [p[1] - q[1], q[0] - p[0]]
            text_pos = (p + q)/2 + 0.1*normal
            po.add(ax.annotate(
                str(round(self.getData().costToParent(), 2)), 
                ((p[0] + q[0])/2, (p[1] + q[1])/2), 
                textcoords="offset points", 
                xytext=text_pos, 
                ha='center'
                ))
        
        if self.getParent().isRoot() and plot_direction:
            dx = q[0] - p[0]
            dy = q[1] - p[1]
            po.add(ax.arrow(
                p[0], 
                p[1], 
                dx, 
                dy, 
                width=0, 
                head_width = 0.02, 
                head_length=0.033, 
                overhang=0.3, 
                length_includes_head=True,**kwargs
                ))
            
        return po


    def getPathToRoot(self) -> np.ndarray:
        """
            Returns the path to the root.
        """
        if self.isRoot():
            return self.getData().p().reshape(2,1)
        else:
            return np.concatenate((
                self.getData().p().reshape(2,1), 
                self.getParent().getPathToRoot()), 
                axis=1
                )

    def plotPathToRoot(
            self, 
            po : PlotObject = None, 
            ax : plt.Axes = None, 
            annotate_cost = False, 
            plot_direction = False, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        """
            Plots the path to the root.
        """
        if self.isRoot() or self is None:
            return po
        
        if po is None:
            po = PlotObject()
        
        po.add(self.plotPathToParent(
            ax, 
            annotate_cost=annotate_cost, 
            plot_direction=plot_direction, 
            **kwargs
            ))
        self.getParent().plotPathToRoot(
            po, 
            ax, 
            annotate_cost=annotate_cost, 
            plot_direction=plot_direction, 
            **kwargs
            )
        return po

    def plotTree(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        queue = [self]
        while queue:
            node = queue.pop(0)
            po.add(node.plotPathToParent(ax, **kwargs))
            queue.extend(node.getChildren())
        return po
