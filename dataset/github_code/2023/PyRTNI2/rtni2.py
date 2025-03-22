from misc import _generate_iterator4reconnections, _WeingartenNote
from sympy import symbols, simplify
import itertools
import copy
import math

try:
    import tensornetwork as tn
except ModuleNotFoundError:
    pass

try:
    import numpy as np
except ModuleNotFoundError:
    pass

##### nodes, tensors, matrices #####

class NodeConnectionError(Exception):
    pass
class TensorOperationError(Exception):
    pass
class TriviallyZero(Exception):
    pass
class RandomTensor(Exception):
    pass
class Integration(Exception):
    pass


class _Node:
    def __init__(self, info, tensor):
        self._info = info
        self._tensor = tensor
        
        self._connected_to = None
        
        self._in_integration_mode = False
    
    # connecting nodes by *.
    def __mul__(self, other):
        if self._connected_to == other:
            if self._in_integration_mode:
                raise NodeConnectionError('These two are already connected to each other.')
            print('These two are already connected to each other.')
            return
        elif self._connected_to != None:
            if other._connected_to != None:
                raise NodeConnectionError('Both nodes have already been connected to some nodes. Disconnect them, first.')
            else:
                raise NodeConnectionError('The first node has already been connected to some node. Disconnect them, first.')   
        elif other._connected_to != None:
            raise NodeConnectionError('The second node has already been connected to some node. Disconnect them, first.')
        
        self._connected_to = other
        other._connected_to = self

        if not self._in_integration_mode:
            print('Connected.')
    
    # disconnecting the specified two nodes by /.
    def __truediv__(self, other):
        if self._connected_to != other:
            if self._in_integration_mode:
                raise NodeConnectionError('No connection between these two nodes.')
            print('No connection between these two nodes.')
            return
        
        self._connected_to = None
        other._connected_to = None
        
        if not self._in_integration_mode:
            print('Disconnected.')
    
    # disconnecting the current node and the other unspecified. 
    def __invert__(self,):
        if self._connected_to == None:
            if self._in_integration_mode:
                raise NodeConnectionError('This node has no connection.')
            print('This node has no connection.')
            return
        else:
            if not self._in_integration_mode:
                print(f'Disconnecting from {self._connected_to._info}.')
                
            self._connected_to._connected_to = None
            self._connected_to = None
            
            if not self._in_integration_mode:
                print('Disconnected.')
            
    def show_info(self, subject):
        info_other = {} if self._connected_to == None else self._connected_to._info
        info = {f'{subject}' : self._info, 'connected': info_other}
        return f'{info}' 
        
    def __repr__(self,):
        return self.show_info('this')
    
    def get_info(self,):
        return f'{self._info}'

    def get_connected_to(self,):
        return self._connected_to

    def get_tensor(self,):
        return self._tensor
    
class Tensor:
    def __init__(self, name, dims, tensor_id=0, nickname = None):
        self._name = name
        self._nickname = name+f'_{tensor_id}' if nickname==None else nickname 
        
        self._transpose = False
        self._conjugate = False
        
        if self.__class__.__name__ == 'Matrix':
            self._dims_mat = (tuple(dims[0]), tuple(dims[1]))
            self._dims = sum(self._dims_mat, ())
        else:
            self._dims = tuple(dims)
        self._tensor_id = tensor_id
        
        self._nodes = [
            _Node(info={'tensor_name': self._name, 'tensor_id': self._tensor_id, 'tensor_nickname': self._nickname,
                       'space_id': i, 'dim': self._dims[i],
                      'is_dangling_end': False}, tensor=self) 
            for i, dim in enumerate(self._dims)]
        
        self._family = [self] if tensor_id == 0 else None
    
    def __repr__(self,):
        return f'{self.get_info()}\n' 

    def __getitem__(self, i):
        return self._nodes[i]

    def __len__(self):
        return len(self._nodes)
    
    def __call__(self, i):
        return self._nodes[i]
    
    def clone(self, nickname=None):
        new_id = max([t._tensor_id for t in self._family]) + 1
        if self.__class__.__name__ == 'Tensor':
            new_tensor = Tensor(name=self._name, dims=self._dims, tensor_id = new_id, nickname = nickname)
        elif self.__class__.__name__ == 'Matrix':
            new_tensor = Matrix(name=self._name, dims=self._dims_mat, tensor_id = new_id, nickname = nickname)
        
        self._family += [new_tensor]
        for t in tuple(self._family):
            t._family = self._family
        
        return new_tensor
    
    def about_nodes(self,):
        for i, node in enumerate(self._nodes):
            print(f'Node {i}: {node}')
        print()
        
    def _nodes_all(self,):
        return self._nodes
        
    def _integration_mode(self,):
        for node in self._nodes:
            node._in_integration_mode = True
    def _input_mode(self,):
        for node in self._nodes:
            node._in_integration_mode = False
            
    def conjugate(self,):
        self._conjugate = bool(True - self._conjugate)
        
    def transpose(self,):
        if self.__class__.__name__ == 'Tensor':
            raise TensorOperationError('The operation transpose is not defined for tensors.')
        self._transpose = bool(True - self._transpose)
        
    def adjoint(self,):
        if self.__class__.__name__ == 'Tensor':
            raise TensorOperationError('The operation adjoint is not defined for tensors.')
        self._conjugate = bool(True - self._conjugate)
        self._transpose = bool(True - self._transpose)
        
    def get_info(self, ):
        return {'tensor_name': self._name, 'tensor_id': self._tensor_id, 'tensor_nickname': self._nickname,
                'dims': self._dims, 'transpose': self._transpose, 'conjugate': self._conjugate}

    def get_nodes(self,):
        return self._nodes

def tensor(name, dims, tensor_id=0, nickname=None):
    return Tensor(name, dims, tensor_id, nickname)


class Matrix(Tensor):
    def __init__(self, name, dims, tensor_id=0, nickname=None):
        super().__init__(name, dims, tensor_id, nickname)
        self.num_out_original = len(self._dims_mat[0])
        self.num_all = self.num_out_original + len(self._dims_mat[1])
        
        for i, node in enumerate(self._nodes):
            if i < self.num_out_original:
                node._info.update({'side_original': 'out', 'side_space_id': i})
            else:
                node._info.update({'side_original': 'in', 'side_space_id': i%self.num_out_original})
            
    def out(self, i):
        if self._transpose:
            return self._nodes[self.num_out_original + i]
        else:
            return self._nodes[i]
    def inn(self, i):
        if self._transpose:
            return self._nodes[i]
        else:
            return self._nodes[self.num_out_original + i]
        
    def _nodes_out(self,):
        return [self._nodes[i] for i in range(self.num_out_original)]
    
    def _nodes_in(self,):
        return [self._nodes[i] for i in range(self.num_out_original, self.num_all)] 
    
    def _nodes_both(self,):
        return [self._outs(), self._inns()] 
        
    def get_info(self, ):
        return {'tensor_name': self._name, 'tensor_id': self._tensor_id, 'tensor_nickname': self._nickname,
                'dims': self._dims, 'dims_mat': self._dims_mat,
                      'transpose': self._transpose, 'conjugate': self._conjugate}
    
def matrix(name, dims, tensor_id=0, nickname=None):
    return Matrix(name, dims, tensor_id, nickname)

# One needs these dangling nodes for integrations. 
class _DanglingTensor:
    def __init__(self, history_dict):
        
        self._name = 'dg_' + history_dict['random_tensor_name'] # tensor_name
        self._info = history_dict.copy()

        self._nodes = []
    
    def __repr__(self,):
        return f'{self.get_info()}\n' 

    def __getitem__(self, i):
        return self._nodes[i]

    def __len__(self):
        return len(self._nodes)
    
    def about_nodes(self,):
        for i, node in enumerate(self._nodes):
            print(f'Node {i}: {node}')
        print()
        
    def get_info(self, ):
        return {'tensor_name': self._name}  

    def get_nodes(self, ):
        return self._nodes   
    
    # nodes in tensors are made with info={'tensor_name': , 'tensor_id': , 'space_id': , 'dim': }
    def add(self, node): # 'tensor_id': 0, 'space_id': 1, 'dim': n
        d = node._info.copy()
        d['is_dangling_end'] = True
        d['tensor_name_origonal'] = d['tensor_name']
        d['tensor_name'] = 'dg_' + d['tensor_name']
        d['tensor_nickname'] = 'dg_' + d['tensor_nickname']
        dangling_node = _Node(info=d, tensor=self)
        dangling_node._in_integration_mode = True
        dangling_node * node
        self._nodes.append(dangling_node)
        
    def _integration_mode(self,):
        pass

##### 



# create an object of tensor-network consisting of tensors. 
class _TensorNetwork:
    def __init__(self, tensors=None, initial_weight=1):
        self._tensors = []
        if tensors != None:
            self._add(tensors)
            
        self._initial_weight = initial_weight
        # history of integration, which will branch. 
        self._history = []
        
        # to access the system in which this tensor-network is.  
        self._system = None

        self._info_removed = {}
        
    def __repr__(self,):
        return f'tensors:\n{self._tensors}'

    def __getitem__(self, i):
        return self._tensors[i]
    
    def __len__(self):
        return len(self._tensors)
    
    ##### building a tensor-network.
    # add tensors into the current tensor-network, at the beginning. 
    def _add(self, tensors):
        if type(tensors) != list:
            tensors = [tensors]
        for t in tensors:
            if t in self._tensors:
                info = t.get_info()
                print(f'tensor {info["tensor_name"]} clone {info["tensor_id"]} is already in the system.')
            else:
                self._tensors.append(t)
                info = t.get_info()
#                 print(info.keys())
                print(f'tensor {info["tensor_name"]} clone {info["tensor_id"]} has been added.')
    
    ##### preparing for integration. 
    # before integration, a clone is to be made individually for different reconnections. 
    def _clone(self,):
        # make a shallow copy to keep track on the system, which this tensor is in.
        original_system = copy.copy(self._system)
        self._system = None
        clone = copy.deepcopy(self)
        # put the copy back into the spot. 
        self._system = original_system
        clone._system = original_system
#         print('clone', id(self._system), id(clone._system))
        
        return clone
    
    # before integration, unconnected nodes of random tensors need to be connected to dummy nodes. 
    def _add_dangling_tensor(self, history_dict):
        # create a dangling tensor for each random tensor name.
        random_tensor_name = history_dict['random_tensor_name']
        dangling_tensor = _DanglingTensor(history_dict) 
        for t in self._get_random_tensors(random_tensor_name):
            for node in t._nodes:
                if node._connected_to == None:
                    dangling_tensor.add(node)
        self._tensors.append(dangling_tensor)    
        
    ##### intractive mode - on and off.     
    # integration mode gives feedbacks, but input mode does not. 
    def _integration_mode(self,):
        for t in self._tensors:
            t._integration_mode()
            
    def _input_mode(self,):
        for t in self._tensors:
            t._input_mode()  
            
            
    ##### getting info about the current tensor-network.
    # get all tensors under conditions. 

    def _get_tensors(self):
        return self._tensors
    
    def _get_random_tensors(self, random_tensor_name):
        return [t for t in self._tensors if (t.__class__.__name__ != '_DanglingTensor') and (t._name == random_tensor_name)]

    def _get_dangling_tensors(self,):
        return [t for t in self._tensors if (t.__class__.__name__ == '_DanglingTensor')]  
   
    # exclude the removed.
    def _get_nondangling_tensors(self,):
        return [t for t in self._tensors if (t.__class__.__name__ != '_DanglingTensor') and (t._name not in self._system._info_removed)]
    
    # get all edges of tensors in the current tensor-network. 
    def _get_edges(self,):
        edges = []
        for t in self._tensors:
            for node in t._nodes:
                if node._connected_to != None:
                    edge = sorted([node._info, node._connected_to._info], # set the order to avoid double-counting. 
                    key = lambda x: (x['tensor_name'], x['tensor_id'], x['space_id'], x['is_dangling_end'])
                                 )
                    if edge not in edges:
                        edges.append(edge)
        return edges
    
    # show all edges of tensors in the current tensor-network. 
    def show_edges(self, counting=False):
        edges = self._get_edges()
        if counting: 
            count = 0                
        for node1, node2 in edges:
            if counting: 
                count += 1; print(count)
            print(node1)
            print('<->')
            print(node2)
            print()
     
    # calculating the weight. 
    def weight(self, dim_symbols_dict=None, side='out'):
        is_symbolic = True if dim_symbols_dict == None else False
#         print('is_symbolic', is_symbolic)
        current_weight =1
        
        # contribution of the initial weight. 
        current_weight *= self._initial_weight
                    
        for h in self._history:
            # contribution of loops.
            for l_dict in h['loops']:
                current_weight *= l_dict['dim'] if is_symbolic else l_dict['dim'].subs(dim_symbols_dict)

#             ### ATTENTION!!!! Under progress. 
#             if not is_symbolic: continue 
                
            # contribution of Weingarten functions.
            if h['yd'] != None:
                yd = h['yd']
                size =  sum(yd)
                dim = math.prod(h['dims_mat'][0]) if side=='out' else  math.prod(h['dims_mat'][1])
                if not is_symbolic:
                    dim = dim.subs(dim_symbols_dict)
                is_complex = h['is_complex']
                current_weight *= self._system._wn.get(
                    is_symbolic=is_symbolic, is_complex=is_complex, size=size, dim=dim, yd=yd)
                        
        return current_weight
    
    def to_tn(self, include_danglings=True):
        return ToTN(self, include_danglings)
    
    
# create an object of parallel tensor-networks with weights. 
# before integration it has only one tensor-network. 
class TensorNetworks:
    def __init__(self, t=None, weight=1):
        
        # all parallel tensor-networks are in this list.
        # at the initialization, an empty tensor-network is created by _Tensornetwork.
        self._tensornetworks = [_TensorNetwork(t, weight)]
        self._tensornetworks[0]._system = self
        
        # info on random tensors integrated; keys are names. 
        self._info_removed = {}
       
        # creating the common instances for weingarten functions and numbers.
        self._wn = _WeingartenNote()
        
        # for errors. 
        self.allowed_types = ['unitary', 'orthogonal', 'real_gaussian', 'complex_gaussian']
     
    def __getitem__(self, i):
        tensornetwork = self._tensornetworks[i]
        tensornetwork._info_removed = self._info_removed
        tensornetwork._tensors = [t for t in tensornetwork._tensors if t._name not in self._info_removed]
        return tensornetwork
    
    def __len__(self):
        return len(self._tensornetworks)
    
    def copy(self,):
        return copy.deepcopy(self)
    
    ##### deal with the tensor-network created at the initialization. 
    # add tensors to that tensor-network. 
    def add(self, tensors):
        if self._info_removed != {}:
            raise Integration('New tensors can be added before integrating a tensor network.')
        self._tensornetworks[0]._add(tensors)
    
    # show the edges of that tensor-network.
    def show_edges(self, counting=False):
        if self._info_removed != {}:
            raise Integration('Specify a tensor network after integrations.')
        self._tensornetworks[0].show_edges()
    
    ##### show all the tensor-networks in the system. 
    def show(self, counting=False, detail=False, dim_symbols_dict=None):
        for tensornetwork in self._tensornetworks:
            if detail:
                print('History of tensor network.')
                print(tensornetwork._history)
                print()
            print('Weight:')
            display(tensornetwork.weight(dim_symbols_dict=dim_symbols_dict))
            print()
            print('Edges:')
            tensornetwork.show_edges(counting)
            print()
    
    ##### integration.
    # gather info for integration.
    def _make_integration_dict(self, random_tensor_name, random_tensor_type):
        integration_dict = {}
        
        # record the basic info.
        integration_dict['random_tensor_name'] = random_tensor_name
        integration_dict['random_tensor_type'] = random_tensor_type
        
        if random_tensor_type not in self.allowed_types:
            raise RandomTensor(f'A type of random tensors must be one of the following {self.allowed_types}')
        
        
        # record if complex or not, and if gourp or not.
        if random_tensor_type in ['unitary', 'complex_gaussian']:
            is_complex = True
        else:
            is_complex = False
            
        if random_tensor_type in ['unitary', 'orthogonal']:
            is_group = True
        else:
            is_group = False
            
        integration_dict['is_complex'] = is_complex
        integration_dict['is_group'] = is_group
        
        
        # use the first tensor-network  as a sample for the necessary info for integration. 
        rts_sample = self._tensornetworks[0]._get_random_tensors(random_tensor_name)
        num_rts = len(rts_sample)
        if num_rts==0:
            raise RandomTensor('There is no such tensors or matrices in the system.')
        
        # get the dim(s).
        if is_group:
            integration_dict['dims_mat'] = rts_sample[0]._dims_mat
            integration_dict['dims_tensor'] = rts_sample[0]._dims
        else:
            integration_dict['dims_tensor'] = rts_sample[0]._dims
        
        # get the nums of random tensors and make sure that the integral is not trivially zero. 
        if is_complex:
            num_vanilla   = sum(1 for t in rts_sample if t._conjugate == False)
            num_conjugate = sum(1 for t in rts_sample if t._conjugate == True)
            if num_vanilla != num_conjugate:
                raise TriviallyZero('The numbers of random tensors and their complex conjugates must match.') 
            integration_dict['size'] = num_vanilla
                
        else:
            if len(rts_sample)%2 !=0:
                raise TriviallyZero('Since the numner of random tensors is odd, the integral vanishes.') 
            integration_dict['size'] = num_rts//2
        
        return integration_dict
        


    # integrate the system wrt the nominated random tensors; specify the type here. 
    def integrate(self, random_tensor_name, random_tensor_type):
        
        # make sure that the system has not been integrated over the nominated. 
        if random_tensor_name not in self._info_removed:
            # make the dict of the necessary info for integration. 
            integration_dict = self._make_integration_dict(random_tensor_name, random_tensor_type)
            # record that the integration is done; not yet though. 
            self._info_removed[random_tensor_name] = integration_dict
        
        # collect new tensor-networks originated from all the tensor-networks in the system.
        new_tensornetworks_all = []
        for tensornetwork in self._tensornetworks:
            # each tensor-network branches while integration the system.
            new_tensornetworks_all += self._integrate_each_tensornetwork(tensornetwork, integration_dict)
            
        # replace the old list of tensor-networks by the new.
        self._tensornetworks = new_tensornetworks_all
        print(f'Integrated. We now have {len(self._tensornetworks)} tensor networks.')
        print()
        
    
    # integrate each tensor-network, which will branch into several. 
    def _integrate_each_tensornetwork(self, tensornetwork, integration_dict):
        # collect new tensor-networks originated from the specified tensor-network.
        new_tensornetworks = []
        
        # create an object which works as "a trunk" for "branches". 
        new_tn_maker = _Reconnection(tensornetwork, integration_dict)
        
        # make an iterator for all possible reconnections. 
        if integration_dict['is_group']:
            iterator4reconnections = itertools.product(
                _generate_iterator4reconnections(integration_dict['size'], True, integration_dict['is_complex']),
                _generate_iterator4reconnections(integration_dict['size'], True, integration_dict['is_complex'])
            )
        else:
            iterator4reconnections = _generate_iterator4reconnections(integration_dict['size'], False, integration_dict['is_complex'])
            
        # iterate by the iterator to reconnect the the specified tensor-network for new tensor-networks. 
        for p in iterator4reconnections:
            reconnection_dict = {'plan': p, 'loops': []}
            new_tensornetworks.append(new_tn_maker._reconnect(reconnection_dict))
            
        return new_tensornetworks

def tensornetworks(t=None, weight=1):
    return TensorNetworks(t=t, weight=weight)
        
    
    
# to make an object out of a tensor-network to reconnect its tensors for all permutations; clone and reconnect. 
# integration process is controled at the higher level. 
class _Reconnection:
    def __init__(self, tensornetwork, integration_dict):
        # take in the target tensor-network. 
        self.tensornetwork = tensornetwork
        
        # make class instances out of the integration dict. 
        for k,v in integration_dict.items():
            setattr(self, k, v)
        
        # to be descended to the children.
        self.integration_dict = integration_dict
        
        # pick the relevant reconnection function.
        self.reconnect_tensor = self._get_reconnect_function(integration_dict['random_tensor_type'])
    
    # each branching and integration occurs based on each reconnection_dict.
    def _reconnect(self, reconnection_dict):
        
        # copy the past info to appnd the new info for this reconnection. 
        history_dict = self.integration_dict.copy() 
        
        # copy the target tensor-network as a branch, and get it ready for this reconnection. 
        tensornetwork_clone = self.tensornetwork._clone()
        tensornetwork_clone._integration_mode()
        tensornetwork_clone._add_dangling_tensor(history_dict)
        
        # getting the relevant random tensors.
        random_tensors = tensornetwork_clone._get_random_tensors(self.random_tensor_name)
        
        # reconnect the branch tensor-network.
        self.reconnect_tensor(random_tensors, reconnection_dict) 
        
        # record the info about the new loops; a temporary dict to a permanent dict.
        history_dict['loops'] = reconnection_dict['loops']
        
        # record the info about the combinations; yd (Young diagram) is needed for Weingarten thing. 
        # we needed two pairings in this case.
        if self.is_group:
            (pairs1, perms1), (pairs2, perms2) = reconnection_dict['plan']
            history_dict['pairs'] = (pairs1, pairs2)
            
            if self.is_complex:
                yd = tuple(sorted([l for l, n in (perms1**-1*perms2).cycle_structure.items() for _ in range(n)], reverse=True))
            else:
                yd = tuple(sorted([l for l, n in (perms1*perms2).cycle_structure.items() for _ in range(n//2)], reverse=True))

            history_dict['yd'] = yd
         
        # we needed only one pairing in this case.
        else:
            pairs, _ = reconnection_dict['plan']
            history_dict['pairs'] = pairs 
            history_dict['yd'] = None
        
        # record the info in the history of the tensor-network. 
        tensornetwork_clone._history.append(history_dict)
        
        # return the newly branched tensor-network, associated to individual reconnections. 
        return tensornetwork_clone

    
    ##### get vanilla or coomplex conjugated tensors.
    def _get_vanilla_tensors(self, tensors):
        return [t for t in tensors if t._conjugate == False]
    def _get_conjugate_tensors(self, tensors):
        return [t for t in tensors if t._conjugate == True]
    
        
    ##### 1) reconnect each node, and record a loop if any. 
    def _reconnect_each(self, node1, node2, reconnection_dict):

        # a loop generated.
        if (node1 == node2._connected_to):
            # recording the node generating the loop. 
            reconnection_dict['loops'].append(node1._info) 
            node1 / node2
        # no loop generated. 
        else:
            node11 = node1._connected_to; node22 = node2._connected_to
            node11 / node1; node22 / node2
            node11 * node22        
            

    ##### 2) reconnect nodes in lists. <- 1)
    # reconnect nodes for pairs of the vanilla and the complex conjugate (complex case).         
    def _reconnect_complex_nodes(self, vanilla_nodes,  conjugate_nodes, pairs, reconnection_dict):
        for index_v, index_c in pairs:
            for node_v, node_c in zip(vanilla_nodes[index_v], conjugate_nodes[index_c]):
                self._reconnect_each(node_v, node_c, reconnection_dict)
    
    # reconnect nodes for pairs (real case). 
    def _reconnect_real_nodes(self, nodes, pairs, reconnection_dict):
        for index_1, index_2 in pairs:
            for node_1, node_2 in zip(nodes[index_1], nodes[index_2]):
                self._reconnect_each(node_1, node_2, reconnection_dict)
                
    
    ##### 3) reconnect nodes for 4 cases; Gaussian tensors have no side, practically. <- 2)
    def _reconnect_complex_gaussian(self, tensors, reconnection_dict):
        tensors_vanilla =   self._get_vanilla_tensors(tensors)
        tensors_conjugate = self._get_conjugate_tensors(tensors)
        
        vanilla_nodes   = [t._nodes_all() for t in tensors_vanilla]
        conjugate_nodes = [t._nodes_all() for t in tensors_conjugate]
        
        pairs, _ = reconnection_dict['plan']
        self._reconnect_complex_nodes(vanilla_nodes, conjugate_nodes, pairs, reconnection_dict)
        
    def _reconnect_real_gaussian(self, tensors, reconnection_dict):
        nodes = [t._nodes_all() for t in tensors]

        pairs, _ = reconnection_dict['plan']
        self._reconnect_real_nodes(nodes, pairs, reconnection_dict)
        
    def _reconnect_unitary(self, tensors, reconnection_dict):
        tensors_vanilla =   self._get_vanilla_tensors(tensors)
        tensors_conjugate = self._get_conjugate_tensors(tensors)
        
        vanilla_nodes_out = [t._nodes_out() for t in tensors_vanilla]
        vanilla_nodes_in  = [t._nodes_in()  for t in tensors_vanilla]
        conjugate_nodes_out = [t._nodes_out() for t in tensors_conjugate]
        conjugate_nodes_in  = [t._nodes_in()  for t in tensors_conjugate]
        
        (pairs_out, _), (pairs_in, _) = reconnection_dict['plan']
        
        self._reconnect_complex_nodes(vanilla_nodes_out, conjugate_nodes_out, pairs_out, reconnection_dict)
        self._reconnect_complex_nodes(vanilla_nodes_in,  conjugate_nodes_in,  pairs_in, reconnection_dict)
        
    def _reconnect_orthogonal(self, tensors, reconnection_dict):
        nodes_out = [t._nodes_out() for t in tensors]
        nodes_in  = [t._nodes_in()  for t in tensors]
        
        (pairs_out, _), (pairs_in, _) = reconnection_dict['plan']
        
        self._reconnect_real_nodes(nodes_out, pairs_out, reconnection_dict)
        self._reconnect_real_nodes(nodes_in,  pairs_in, reconnection_dict)  
    
    
    ##### pick the proper reconnection function. 
    def _get_reconnect_function(self, random_tensor_type):
        if random_tensor_type == 'complex_gaussian':
            return self._reconnect_complex_gaussian
        if random_tensor_type == 'real_gaussian':
            return self._reconnect_real_gaussian
        if random_tensor_type == 'unitary':
            return self._reconnect_unitary
        if random_tensor_type == 'orthogonal':
            return self._reconnect_orthogonal
        
        
##### translating into TensorNetwork.
# one class instance for each tensor-network.
class ToTN:
    def __init__(self, tensornetwork, include_danglings = True):
        self._tensornetwork = tensornetwork
        self._include_danglings = include_danglings
        
        self._tns_connection_dict, self._tns_dict, self._tns_nickname_dict = self._translate()
        self._connect_tn()
        
        self._tensors_dict = {}  
        
        self._is_nickname_mode = False
        
    
    def subs(self, tensors_dict):
        # make a dictionary first.
        self._tensors_dict.update(tensors_dict)
        # replace the provisional empty arraes with those of interest. 
        for name, tensor_np in tensors_dict.items():
            for tensor_tn in self._tns_dict[name].values():
                tensor_tn.tensor = tensor_np
                
    def get_tns(self):
        return self._tns_dict
    
    def get_tns_nickname(self):
        return  self._tns_nickname_dict
    
    def to_nickname(self):
        if self._is_nickname_mode:
            return
        else:
            self._is_nickname_mode=True
            
        for nickname, node in self._tns_nickname_dict.items():
            if type(node)==dict:
                for i, small_node in node.items():
                    small_node._name = f'{nickname}_{i}'
            else:
                node._name = nickname
                
    def from_nickname(self):
        if not self._is_nickname_mode:
            return
        else:
            self._is_nickname_mode = False
            
        for name, nodes in self._tns_dict.items():
            for i, node in nodes.items():
                if type(node)==dict:
                    for j, small_node in node.items():
                        small_node._name = f'{name}_{i}_{j}'
                else:
                    node._name = f'{name}_{i}'
        
    
    def get_tns_list(self):
        tns_list = []
        for name, copies in self._tns_dict.items():
            for node_tn in copies.values():
                if type(node_tn) == dict:
                    tns_list += list(node_tn.values())
                else:
                    tns_list.append(node_tn)
        return tns_list           
    
    
    ### translate from rtni to tn.
    def _translate(self,):
        
        # for connections in self._connect_tn
        tns_connection_dict = {}
        # to access tensors. 
        tns_dict = {}
        # to rename tensors.
        tns_nickname_dict = {}        

        # translate a tensor.
        for tensor in self._tensornetwork._get_nondangling_tensors():
            name = tensor._name
            nickname = tensor._nickname 
            tensor_id = tensor._tensor_id
            dims_num = len(tensor._dims)
            
            # make a new tn for each tensor.
            new_tn = tn.Node(np.empty([1 for _ in range(dims_num)]), name=f'{name}_{tensor_id}')

            tns_connection_dict.setdefault(name, {}) 
            tns_dict.setdefault(name, {})
            
            # dict[key][id] gives a limb because dict[key] is a tn.
            tns_connection_dict[name][tensor_id] = new_tn
            tns_dict[name][tensor_id] = new_tn
            # for nickname ui. 
            tns_nickname_dict[nickname] = new_tn

        if self._include_danglings:
            for tensor in self._tensornetwork._get_dangling_tensors():

                name = tensor._name
                tns_dict.setdefault(name, {})
                tns_connection_dict.setdefault(name, {})

                # iterate over all nodes under the name; all copies are mixed for danglings. 
                for node in tensor._nodes:

                    tensor_id = node._info['tensor_id']
                    space_id = node._info['space_id']
                    nickname = node._info['tensor_nickname']

                    # make a new tensor for each dangling.
                    new_tn_limb = tn.Node(np.empty(1), name=f'{name}_{tensor_id}_{space_id}')

                    tns_connection_dict[name].setdefault(tensor_id, {})
                    tns_dict[name].setdefault(tensor_id, {})
                    
                    # dict[key][id] gives a limb differently from the tensor case. 
                    tns_connection_dict[name][tensor_id][space_id] = new_tn_limb[0]
                    # there may be more-than-one limbs, i.e. tns. 
                    tns_dict[name][tensor_id][space_id] = new_tn_limb
                    
                    tns_nickname_dict.setdefault(nickname, {})
                    tns_nickname_dict[nickname][space_id] = new_tn_limb
                             
        return tns_connection_dict, tns_dict, tns_nickname_dict
    
    def _connect_tn(self,):
        for edge in self._tensornetwork._get_edges():
            limb0, limb1 = edge
            if not self._include_danglings:
                if limb0['tensor_name'].startswith('dg_') or limb1['tensor_name'].startswith('dg_'):
                    continue
            first  = self._tns_connection_dict[limb0['tensor_name']][limb0['tensor_id']][limb0['space_id']]
            second = self._tns_connection_dict[limb1['tensor_name']][limb1['tensor_id']][limb1['space_id']]
            first ^ second
            
