
# This is a class that contains a lot of functions useful for performing
# checking of how the gradient is flowing inside a network.

import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D 

'''
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function 
'''

'''Plots the gradients flowing through different layers in the net during training.
Can be used for checking for possible gradient vanishing / exploding problems.

Usage: Plug this function in Trainer class after loss.backwards() as 
"plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''



def plot_and_save_grad_flow(named_parameters, fileName):
    
    layers, ave_grads, max_grads = calculateGrads(named_parameters)
    
    plotGrads(layers, max_grads)
    plt.savefig(fileName + '_max.png')
    
    plotGrads(layers, ave_grads)
    plt.savefig(fileName + '_average.png')
    
    return

def plot_and_save_grad_flow_together(named_parameters, fileName):
    
    layers, ave_grads, max_grads = calculateGrads(named_parameters)
    plotGradsTogether(layers, max_grads, ave_grads)
    plt.savefig(fileName)
    
    return

def plotGradsTogether(layers, max_grads, ave_grads):
    
    max_grad_value = np.asarray(max_grads).max()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=max_grad_value) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    return

def plotGrads(layers, grads):
    
    max_grad_value = np.asarray(grads).max()
    plt.bar(np.arange(len(grads)), grads, alpha=0.1, lw=1, color="c")
    plt.hlines(0, 0, len(grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(grads))
    plt.ylim(bottom = -0.001, top=max_grad_value) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['grads', 'zero-gradient'])
    
    return

def calculateGrads(named_parameters):
    
    ave_grads = []
    max_grads= []
    layers = []
    
    plt.close('all')
    
    for n, p in named_parameters:

        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    
    return layers, ave_grads, max_grads
    
    
    #######################################################################
    # Functions taken from:
    # https://github.com/t-vi/pytorch-tvmisc/blob/master/visualize/bad_grad_viz.ipynb
        
'''
def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

x = torch.randn(10, 10, requires_grad=True)
y = torch.randn(10, 10, requires_grad=True)

z = x / (y * 0)
z = z.sum() * 2
get_dot = register_hooks(z)
z.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable
'''