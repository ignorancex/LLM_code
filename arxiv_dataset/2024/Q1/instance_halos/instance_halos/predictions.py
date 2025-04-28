import os

basedir = os.path.dirname(__file__) + "/.."

import numpy as np
import functools
import tensorflow as tf
import logging
import time

import networkx as nx

from . import clustering

from .clustering import Timer

# -------------------------------------------------------------------------------------------- #
# ---------------------------------------- Setup tools --------------------------------------- #
# -------------------------------------------------------------------------------------------- #

def limit_tensorflow_GPU_memory(GPU_percent_mem_use, GPU_total_memory=45556):
    
    import tensorflow as tf
    # GPU_total_memory must be provided in MiB
    GPU_total_memory *= 1.04858 #MiB to MB (necessary to set tensorflow memory limit)
    
    print('Limit GPU memory : ', GPU_total_memory*GPU_percent_mem_use, 'MB / ', GPU_total_memory, 'MB')
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
    #     print("Physical GPUs = ", len(physical_gpus))
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit = GPU_percent_mem_use * GPU_total_memory # In MB
                    )] 
                )
    #             print(gpu)
            except RuntimeError as e:
                print(e)
    # ------- This would be used to check logical GPUs and limit their memory ------- #
    # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    # if logical_gpus:
    #     print("Logical GPUs = ", len(logical_gpus))
    #return 0
    
# -------------------------------------------------------------------------------------------- #
# ----------------------------------------- Data tools --------------------------------------- #
# -------------------------------------------------------------------------------------------- #

def periodic_crop_1d(field, i0, n, axis=0):
    assert n <= field.shape[axis], ("crop %d bigger than domain %d not implemented" % (n, field.shape[axis]))
    
    fs = []
    
    i0 = i0 % field.shape[axis]
    
    for offset in 0, -field.shape[axis]:
        left = np.clip(i0 + offset, 0, field.shape[axis])
        right = np.clip(i0 + offset + n, 0, field.shape[axis])
        
        sl = []
        for i in range(0, len(field.shape)):
            if i == axis:
                sl.append(slice(left, right))
            else:
                sl.append(slice(0, field.shape[i]))
    
        fs.append(field[tuple(sl)])
    
    return np.concatenate(fs, axis=axis)

def periodic_crop(field, i0, n):
    fcrop = field
    
    n = n * np.ones_like(i0)
    
    for i in range(0, len(i0)):
        fcrop = periodic_crop_1d(fcrop, i0[i], n[i], axis=i)
    
    return fcrop

def get_single_crop(data, crop_center, L_half_crop):
    return periodic_crop(data, crop_center - L_half_crop, L_half_crop*2)

def get_single_crop_old(data, crop_center, L_half_crop, crop_mask):
    roll = np.zeros(len(crop_center)).astype(np.int32)
    for ii in range(len(crop_center)):
        roll[ii] = -(crop_center[ii] - L_half_crop)
    roll = tuple(roll)
    
    rolled = np.roll(data, roll, axis=np.arange(len(crop_center)))
    cropped = rolled[crop_mask]
    
    return cropped

def get_crop_centers(LL, NN_crops_per_side, offset=None):
    if offset == None:
        offset = int(LL / NN_crops_per_side / 2)
    crop_centers = []
    for iix in range(NN_crops_per_side):
        for iiy in range(NN_crops_per_side):
            for iiz in range(NN_crops_per_side):
                crop_centers.append((iix*int(LL/NN_crops_per_side)+offset, iiy*int(LL/NN_crops_per_side)+offset, iiz*int(LL/NN_crops_per_side)+offset))
    
    return crop_centers

def define_inputs_for_model(model, delta=None, potential=None, ngrid_out=64, offset=None, semantic=None, append_lag_pos=False, get_output_mask=None, include_semantic_as_input_field=True):
    timer = Timer()
    
    if delta is not None:
        assert delta.shape[0] == delta.shape[1] == delta.shape[2]
        ngrid = delta.shape[0]
        assert ngrid % ngrid_out == 0, "shape of input (%d) has to be multiple of %d" % (delta.shape[0], ngrid_out)
    if potential is not None:
        assert potential.shape[0] == potential.shape[1] == potential.shape[2]
        ngrid = potential.shape[0]
        assert ngrid % ngrid_out == 0, "shape of input (%d) has to be multiple of %d" % (potential.shape[0], ngrid_out)
    
    # crop_input_mask = (slice(0, model.input_shape[1]),)*3
    # crop_output_mask = (slice(0, model.output_shape[1]),)*3
    
    crop_centers = get_crop_centers(ngrid, ngrid // ngrid_out, offset=offset)
    fields = []
    if delta is not None:
        # deltas = np.array([get_single_crop_old(delta, cent, model.input_shape[1]//2, crop_input_mask) for cent in crop_centers])
        deltas = np.array([get_single_crop(delta, np.array(cent), model.input_shape[1]//2) for cent in crop_centers])
        fields.append(deltas)
        timer.logdt("Loaded fields: delta")
    if potential is not None:
        # potentials = np.array([get_single_crop_old(potential, cent, model.input_shape[1]//2, crop_input_mask) for cent in crop_centers])
        potentials = np.array([get_single_crop(potential, np.array(cent), model.input_shape[1]//2) for cent in crop_centers])
        fields.append(potentials)
        timer.logdt("Loaded fields: potential")
    if semantic is not None:
        # semantics = np.array([get_single_crop_old(semantic, cent, model.input_shape[1]//2, crop_input_mask) for cent in crop_centers])
        semantics = np.array([get_single_crop(semantic, np.array(cent), model.input_shape[1]//2) for cent in crop_centers])
        fields.append(semantics)
        timer.logdt("Loaded fields: semantic")
    if append_lag_pos:
        qs = np.zeros(((ngrid // ngrid_out)**3,) + model.input_shape[1:4] + (3,))
        qi = np.arange(0, model.input_shape[1], dtype=np.float32)
        qs[...,0], qs[...,1], qs[...,2] = np.meshgrid(qi, qi, qi, indexing='ij')
        fields.extend([qs[...,0], qs[...,1], qs[...,2]])
        timer.logdt("Loaded fields: lag_pos")
        
    fields = np.stack(fields, axis=-1)
    
    if fields.shape[-1] != model.input_shape[-1]:
        raise ValueError("Something went wrong with the input shapes not matching (maybe forgot potential?)", fields.shape, model.input_shape)
        
    if get_output_mask is not None:
        # mask = np.array([get_single_crop_old(get_output_mask, cent, model.output_shape[1]//2, crop_output_mask) for cent in crop_centers])
        mask = np.array([get_single_crop(get_output_mask, np.array(cent), model.output_shape[1]//2) for cent in crop_centers])
        timer.logdt("Generated output_mask")
        return fields, mask
    else:
        return fields


def reduce_labeling(data):
    uq, inv = np.unique(data, return_inverse=True)
    assert uq[0] >= 0, ("negative label found %d " % uq[0])

    data = np.zeros_like(data)
    arange = np.arange(len(uq))
    
    if uq[0] != 0:
        arange += 1
    data.flat = arange[inv]
    
    return data

# -------------------------------------------------------------------------------------------- #
# ---------------------------------------- Models avail -------------------------------------- #
# -------------------------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=2)
def load_model(model_name):
    if model_name in ("semantic_v1.0", "semantic_v1.0_only_density", "semantic_v1.0_only_potential"):
        model = tf.keras.models.load_model("%s/instance_halos_models/%s.keras" % (basedir, model_name))
        config = dict(ngrid_suggest=64)
        return model, config
    if model_name in ("instance_v1.0", "instance_v1.0_only_density", "instance_v1.0_only_potential", "instance_v1.1_only_potential"):
        model = tf.keras.models.load_model("%s/instance_halos_models/%s.keras" % (basedir, model_name))
        config = dict(ngrid_suggest=64, sem_thresh=0.589, combine_thresh=0.5, append_lag_pos=True, include_semantic_as_model_input=False)
        return model, config
    else:
        raise ValueError("Unknown model %s" % model_name)

# -------------------------------------------------------------------------------------------- #
# -------------------------------------- Semantic model -------------------------------------- #
# -------------------------------------------------------------------------------------------- #

def semantic_model_predictions(model, delta=None, potential=None, ngrid_out=64):
    if (model.output_shape[-1] == 2) and (potential is None):
        raise ValueError("With this model, please also provide the potential as input")
        
    timer = Timer()

    fields = define_inputs_for_model(model, delta=delta, potential=potential, ngrid_out=ngrid_out)
    timer.logdt("Defined Inputs")

    # semantics = model.predict(fields, batch_size=2)
    semantics = np.zeros((fields.shape[0],) + model.output_shape[1:-1])
    for ii in range(fields.shape[0]):
        timer.logdt("Semantic crop "+str(ii+1)+"/"+str(fields.shape[0]))
        semantics[ii:ii+1] = model.predict(fields[ii:ii+1], batch_size=1)[..., 0]
    
    timer.logdt("Finished Semantic Predictions")
    
    ngrid_mod = model.output_shape[1]
    lim = (ngrid_mod//2 - ngrid_out//2, ngrid_mod//2 + ngrid_out//2)
    
    semantic = assemble_cube_grid(semantics[:, lim[0]:lim[1], lim[0]:lim[1], lim[0]:lim[1]])
    timer.logdt("Assembled outputs")
    
    return semantic

def semantic_prediction(delta=None, potential=None, model_name="semantic_v1.0", ngrid_out=None):
    model, config = load_model(model_name)
    if ngrid_out == None:
        ngrid_out=config["ngrid_suggest"]
    semantic = semantic_model_predictions(model=model, delta=delta, potential=potential, ngrid_out=ngrid_out)
    return semantic

def assemble_cube_grid(YYY_pred):
    NN = np.int64(np.cbrt(YYY_pred.shape[0]))
    assert YYY_pred.shape[0] == NN**3, "Can only handle cubic grids"
    
    SS = NN * YYY_pred.shape[1]
    
    out = np.zeros(3*(NN, int(SS/NN)))
    for iix in range(NN):
        for iiy in range(NN):
            for iiz in range(NN):
                tmp_crop = YYY_pred[NN*NN*iix+NN*iiy+iiz]
                out[iix, :, iiy, :, iiz, :] = tmp_crop
    out = np.reshape(out, 3*(SS,))
    return out
    
# -------------------------------------------------------------------------------------------- #
# -------------------------------------- Instance model -------------------------------------- #
# -------------------------------------------------------------------------------------------- #

def assemble_labeled_cube_grid(lattice_crops):
    NN = np.int64(np.cbrt(lattice_crops.shape[0]))
    assert lattice_crops.shape[0] == NN**3, "Can only handle cubic grids"
    
    SS = NN * lattice_crops.shape[1]
    
    dummy_label_counter = 0
    lattice = np.zeros(3*(NN, int(SS/NN))).astype(np.int32)
    for iix in range(NN):
        for iiy in range(NN):
            for iiz in range(NN):
                tmp_crop = lattice_crops[NN*NN*iix+NN*iiy+iiz]
                
                # reduce the label indexing
                tmp_crop = reduce_labeling(tmp_crop)
                
                # displace labels to avoid overlap
                tmp_mask =  tmp_crop != 0
                tmp_crop[tmp_mask] = tmp_crop[tmp_mask] + dummy_label_counter

                # save into the global array
                lattice[iix, :, iiy, :, iiz, :] = tmp_crop

                # increase the label displacement
                dummy_label_counter = np.max(tmp_crop)

    lattice = np.reshape(lattice, 3*(SS,)).astype(np.int32)
    
    return lattice

def fake_clustering(scatter_points):
    return np.random.randint(1,10, size=scatter_points.shape[0])

def independent_crops_instance_model_predictions(model, semantic, delta=None, potential=None, ngrid_out=64, append_lag_pos=True, sem_thresh=0.589, offset=None, debug=False, cluster="meshfree", include_semantic_as_model_input=False, **kwargs):
    
    if offset is None:
        offset = ngrid_out//2
    
    timer = Timer()
    if include_semantic_as_model_input:
        fields, mask = define_inputs_for_model(model, delta=delta, potential=potential, semantic=semantic, 
                                          ngrid_out=ngrid_out, append_lag_pos=append_lag_pos,
                                          get_output_mask=semantic>sem_thresh, offset=offset)
    else:
        fields, mask = define_inputs_for_model(model, delta=delta, potential=potential, 
                                          ngrid_out=ngrid_out, append_lag_pos=append_lag_pos,
                                          get_output_mask=semantic>sem_thresh, offset=offset)
    timer.logdt("independent_crops_instance_model_predictions: Defined Inputs")
    
    # pseudos = model.predict(fields, batch_size=2)
    pseudos = np.zeros((fields.shape[0],) + model.output_shape[1:])
    for ii in range(fields.shape[0]):
        timer.logdt("Pseudo crop "+str(ii+1)+"/"+str(fields.shape[0]))
        pseudos[ii:ii+1] = model.predict(fields[ii:ii+1], batch_size=1)
                
    timer.logdt("independent_crops_instance_model_predictions: Predicted pseudo space")
    
    pred_crops = np.zeros(pseudos.shape[:-1])
    
    if debug:
        imax = 1
    else:
        imax = len(pseudos)
    
    for ii in range(0, imax):
        timer.logdt("Clustering instance crop "+str(ii+1)+"/"+str(imax))
        if cluster == "fake":
            pred_crops[ii,mask[ii]] = fake_clustering(pseudos[ii,mask[ii]])
        elif cluster == "meshfree":
            pred_crops[ii,mask[ii]] = 1+clustering.meshfree_descending_clustering(pseudos[ii,mask[ii]], **kwargs)
        
        timer.logdt("independent_crops_instance_model_predictions: Clustering %d/%d" % (ii+1, len(pseudos)))
        
    if debug:
        return pseudos, pred_crops, mask
    
    ngrid_mod = pred_crops.shape[1]
    lim = (ngrid_mod//2 - ngrid_out//2, ngrid_mod//2 + ngrid_out//2)

    instance_grid = assemble_labeled_cube_grid(pred_crops[:, lim[0]:lim[1], lim[0]:lim[1], lim[0]:lim[1]])
    timer.logdt("Assemble Labeled Grid")

    instance_grid = np.roll(instance_grid, (offset-ngrid_out//2, offset-ngrid_out//2, offset-ngrid_out//2), axis=(0, 1, 2))

    return instance_grid


def combine_instance_labeled_lattices(labels1, labels2, combine_thresh=0.5, quadrant_size=32, unmerged_mode="size"):
    
    timer = Timer()
    
    nlabels1, nlabels2 = np.max(labels1)+1, np.max(labels2)+1
    
    # Create a Graph which defines which labels should be merged
    timer.logdt("Start creating graph")
    G = nx.Graph()
    G.add_nodes_from(np.arange(nlabels1))
    timer.logdt("Finish add_nodes_from nlabels1")
    G.add_nodes_from(nlabels1 + np.arange(nlabels2))
    timer.logdt("Finish add_nodes_from nlabels2")
    
    ngrid = labels1.shape[0]
    
    def link_quadrant(imin, qsize):
        l1sel = labels1[imin[0]:imin[0]+qsize, imin[1]:imin[1]+qsize, imin[2]:imin[2]+qsize]
        l2sel = labels2[imin[0]:imin[0]+qsize, imin[1]:imin[1]+qsize, imin[2]:imin[2]+qsize]
        
        size1 = np.bincount(l1sel.flatten())
        size2 = np.bincount(l2sel.flatten())
        
        possible_links, counts = np.unique(np.stack((l1sel.flatten(), l2sel.flatten()), axis=-1), axis=0, return_counts=True)

        for i,(link,intersection_size) in enumerate(zip(possible_links, counts)):
            if (link[0] == 0) | (link[1] == 0):
                continue
            
            # Add a merging edge to the graph if the intersection over union is large enough
            union_size = size1[link[0]] + size2[link[1]] - intersection_size
            
            if intersection_size >= combine_thresh*union_size:
                G.add_edge(link[0], nlabels1+link[1])
    
    for i in range(0, ngrid//quadrant_size):
        timer.logdt("link_quadrant i "+str(i+1)+"/"+str(ngrid//quadrant_size))
        for j in range(0, ngrid//quadrant_size):
            for k in range(0, ngrid//quadrant_size):
                link_quadrant((i*quadrant_size,j*quadrant_size,k*quadrant_size), quadrant_size)

    # Create a nodemap that maps each of the original nodes to a new id that
    # is unique for each connected component in the graph
    nodemap = np.zeros(nlabels1+nlabels2, dtype=np.int64)
    
    idoffset = 1
    for c in nx.connected_components(G):
        for node in c:
            nodemap[node] = idoffset
        
        idoffset += 1
    
    # Explicitly map the background to 0
    nodemap[0] = nodemap[nlabels1] = 0
    
    # Create a map with the new ids
    new_id = np.zeros_like(labels1)
    
    # For merged labels the ids defined in both maps should point
    # to the same new id
    merge_mask = nodemap[labels1] == nodemap[labels2 + nlabels1]
    new_id[merge_mask] = nodemap[labels1[merge_mask]]
    
    size1 = np.bincount(labels1.flatten())
    size2 = np.bincount(labels2.flatten())

    # For unmerged labels we take the id of the larger group
    # that resides at each location.
    if unmerged_mode == "newsize":
        newsize = np.bincount(new_id.flatten(), minlength=np.max(nodemap)+1)

        newsize1 = np.max([size1, newsize[nodemap[np.arange(0, nlabels1)]]], axis=0)
        newsize2 = np.max([size2, newsize[nodemap[np.arange(nlabels1, nlabels1+nlabels2)]]], axis=0)

        sel1 = newsize1[labels1[~merge_mask]] >= newsize2[labels2[~merge_mask]]
    elif unmerged_mode == "size":
        sel1 = size1[labels1[~merge_mask]] >= size2[labels2[~merge_mask]]
    else:
        raise ValueError("Unknown mode %s" % unmerged_mode)
    
    new_id[~merge_mask] = nodemap[labels1[~merge_mask]] * sel1 + nodemap[nlabels1+labels2[~merge_mask]] * (~ sel1)
    
    return new_id

def instance_model_predictions(model, semantic, delta=None, potential=None, ngrid_out=64, append_lag_pos=True, sem_thresh=0.589, combine_thresh=0.5, include_semantic_as_model_input=False, **kwargs):
    timer = Timer()
    kwargs = dict(model=model, semantic=semantic, delta=delta, potential=potential, ngrid_out=ngrid_out, 
                  append_lag_pos=append_lag_pos, sem_thresh=sem_thresh, include_semantic_as_model_input=include_semantic_as_model_input, **kwargs)
    lattice1 = independent_crops_instance_model_predictions(**kwargs, offset=None)
    timer.logdt("instance_model_predictions: Predicted Lattice 1")
    lattice2 = independent_crops_instance_model_predictions(**kwargs, offset=0)
    timer.logdt("instance_model_predictions: Predicted Lattice 2")
    
    instance = combine_instance_labeled_lattices(lattice1, lattice2, quadrant_size=ngrid_out//2, combine_thresh=combine_thresh)
    timer.logdt("instance_model_predictions: Combine Lattices")

    return instance

def instance_prediction(semantic, delta=None, potential=None, model_name="instance_v1.0", ngrid_out=None, **kwargs):
    model, config = load_model(model_name)
    if ngrid_out == None:
        ngrid_out=config["ngrid_suggest"]
    kwargs = dict(delta=delta, potential=potential, ngrid_out=ngrid_out, **kwargs)
    instance = instance_model_predictions(model, semantic, **kwargs)
    return instance

# -------------------------------------------------------------------------------------------- #
# -------------------------------------- Panoptic model -------------------------------------- #
# -------------------------------------------------------------------------------------------- #

def panoptic_prediction(delta=None, potential=None, model_semantic="semantic_v1.0", model_instance="instance_v1.0", get_semantic=True, ngrid_out=None, **kwargs):
    timer = Timer()
    semantic = semantic_prediction(model_name=model_semantic, delta=delta, potential=potential, ngrid_out=ngrid_out)
    timer.logdt("panoptic_prediction: Semantic Predictions")
    kwargs = dict(delta=delta, potential=potential, ngrid_out=ngrid_out, **kwargs)
    instance = instance_prediction(semantic, model_name=model_instance, **kwargs)
    timer.logdt("panoptic_prediction: Instance Predictions")

    if get_semantic:
        return semantic, instance
    else:
        return instance
