import numpy as np

def block_patching(w_j, L_next, assignment_j_c, layer_index, model_meta_data, 
                                fc_outshape, 
                                layer_type="fc", 
                                dataset="cifar10",
                                network_name="lenet"):
    """
    In CNN, weights patching needs to be handled block-wisely
    We handle all conv layers and the first fc layer connected with the output of conv layers here
    """
    #print('--'*15)
    #print("ori w_j shape: {}".format(w_j.shape))
    #print("input layer_type: {}".format(layer_type))
    #print("L_next: {}".format(L_next))
    #print("assignment_j_c: {}, length of assignment: {}".format(assignment_j_c, len(assignment_j_c)))
    #print("correspoding meta data: {}".format(model_meta_data[2 * layer_index - 2]))
    #print("layer index: {}".format(layer_index))
    #print('--'*15)
    if assignment_j_c is None:
        return w_j

    layer_meta_data = model_meta_data[2 * layer_index - 2]
    prev_layer_meta_data = model_meta_data[2 * layer_index - 2 - 2]

    if layer_type == "conv":    
        new_w_j = np.zeros((w_j.shape[0], L_next*(layer_meta_data[-1]**2)))

        # we generate a sequence of block indices
        block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(L_next)]
        ori_block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(layer_meta_data[1])]
        for ori_id in range(layer_meta_data[1]):
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]
        assignment_j_c.sort()
        rematch_localblocks = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in assignment_j_c]
        w = list(np.array(rematch_localblocks).flatten())
        new_w = new_w_j[:, w]
    elif layer_type == "fc":
        # replace the estimated_output.view(-1).size()[0]
        # with your own computation of len of conv output vector
        new_w_j = np.zeros((w_j.shape[0],  int(np.prod(fc_outshape[1:]))))
#         print("estimated_output shape : {}".format(fc_outshape[1:]))
#         print("meta data of previous layer: {}".format(prev_layer_meta_data))
        
        block_indices = [np.arange(i* fc_outshape[1]**2, (i+1)* fc_outshape[1]**2) for i in range(L_next)]
        #for i, bid in enumerate(block_indices):
        #    print("{}, {}".format(i, bid))
        #print("**"*20)
        ori_block_indices = [np.arange(i* fc_outshape[1]**2, (i+1)* fc_outshape[1]**2) for i in range(prev_layer_meta_data[0])]
        #for i, obid in enumerate(ori_block_indices):
        #    print("{}, {}".format(i, obid))
        #print("assignment c: {}".format(assignment_j_c))
        for ori_id in range(prev_layer_meta_data[0]):
            #print("{} ------------ to ------------ {}".format(block_indices[assignment_j_c[ori_id]], ori_block_indices[ori_id]))
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]
        assignment_j_c.sort()
        rematch_localblocks = [np.arange(i* fc_outshape[1]**2, (i+1)* fc_outshape[1]**2) for i in assignment_j_c]
        w = list(np.array(rematch_localblocks).flatten())
        new_w = new_w_j[:, w]
    
    return new_w
