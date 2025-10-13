import torch

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune
import tensorly as tl
import gc
from tensorly.decomposition import parafac, tucker, tensor_train, partial_tucker


class GPTJLaser(AbstractLaser):
    n_heads = 16
    hidden_size = 4096
    head_dim = hidden_size // n_heads

    def __init__(self):
        super(AbstractLaser, self).__init__()

    @staticmethod
    def convert_name(name):
        """
        Convert a given generic transformer layer name to a model specific layer name(s)
        :param name: generic name
        :return: model specific layer name(s)
        """

        ''' 
            For a given layer, we can modify the following type individually or all at onces
    
            transformer.h.26.ln_1.weight
            transformer.h.26.ln_1.bias
            transformer.h.26.attn.k_proj.weight     -> k_proj
            transformer.h.26.attn.v_proj.weight     -> v_proj
            transformer.h.26.attn.q_proj.weight     -> q_proj
            transformer.h.26.attn.out_proj.weight   -> out_proj
            transformer.h.26.mlp.fc_in.weight       -> fc_in
            transformer.h.26.mlp.fc_out.weight      -> fc_out
        '''

        if name == "k_proj":
            converted_names = "attn.k_proj.weight"
        elif name == "q_proj":
            converted_names = "attn.q_proj.weight"
        elif name == "v_proj":
            converted_names = "attn.v_proj.weight"
        elif name == "out_proj":
            converted_names = "attn.out_proj.weight"
        elif name == "fc_in":
            converted_names = "mlp.fc_in.weight"
        elif name == "fc_out":
            converted_names = "mlp.fc_out.weight"
        elif name == "None":
            converted_names = "None"
        elif name == "mlp":
            converted_names = ["mlp.fc_in.weight", "mlp.fc_out.weight"]
        elif name == "attn":
            converted_names = ["attn.k_proj.weight", "attn.q_proj.weight", "attn.v_proj.weight", "attn.out_proj.weight"]
        elif name == "all":
            converted_names = ["attn.k_proj.weight", "attn.q_proj.weight", "attn.v_proj.weight",
                               "attn.out_proj.weight", "mlp.fc_in.weight", "mlp.fc_out.weight"]
        else:
            raise AssertionError(f"Unhandled name {name}")

        return converted_names

    @staticmethod
    def _modify_layer(name, lnum_to_modify, lname_to_modify, converted_names):

        # Check for layer number match
        # If must be either -1 meaning modify all layers, or must match the given layer number
        if lnum_to_modify != -1 and not name.startswith(f"transformer.h.{lnum_to_modify}."):
            return False

        # Check if layer type needs to be modified.
        #      'all', 'mlp', 'attn', 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out'
        # If all, then modify all
        # If mlp, then only MLP
        # If attn, then only attn
        # Otherwise, update a given layer type

        if type(converted_names) == list:
            modify_flag = any([name.endswith(f"{converted_name}") for converted_name in converted_names])
        elif type(converted_names) == str:
            modify_flag = name.endswith(f"{converted_names}")
        else:
            raise AssertionError(f"Type should be list or str. Found {type(converted_names)}.")

        return modify_flag
    
    @staticmethod
    def get_3D_Tucker_edited_model(model, lnum, device, qkvo_rank, attention_matrix, in_place=True):        
        if attention_matrix == 'Q':
            print('Extracting weight Q')
            tensor = model.transformer.h[lnum].attn.q_proj.weight.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim)
        elif attention_matrix == 'K':
            print('Extracting weight K')
            tensor = model.transformer.h[lnum].attn.k_proj.weight.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim)
        elif attention_matrix == 'V':
            print('Extracting weight V')
            tensor = model.transformer.h[lnum].attn.v_proj.weight.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim)
        elif attention_matrix == 'O':
            print('Extracting weight O')
            tensor = model.transformer.h[lnum].attn.out_proj.weight.T.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim)
        
        tl.set_backend('pytorch')
        # shape = [hidden_size, n_heads, head_dim]
        tensorly_tensor = tl.tensor(tensor, device=device)
        
        print("=" * 50)            
        print('Partial Tucker decomposition')

        assert qkvo_rank <= GPTJLaser.head_dim, f"rank exceeds head_dim. head_dim={GPTJLaser.head_dim}, rank={qkvo_rank}"
        (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2], rank=[qkvo_rank*GPTJLaser.n_heads, qkvo_rank], init='svd', tol=1e-5, verbose=True)
        reconstructed_tensor = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1]], modes=[0, 2])

        reconstruction_error = torch.norm(reconstructed_tensor - tensorly_tensor) / torch.norm(tensorly_tensor)
        print(f'Reconstruction error: {reconstruction_error}')
        
        if attention_matrix == 'Q':
            print('Updating weight Q')
            model.transformer.h[lnum].attn.q_proj.weight = torch.nn.Parameter(reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size))
        elif attention_matrix == 'K':
            print('Updating weight K')
            tensor = model.transformer.h[lnum].attn.k_proj.weight = torch.nn.Parameter(reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size))
        elif attention_matrix == 'V':
            print('Updating weight V')
            model.transformer.h[lnum].attn.v_proj.weight = torch.nn.Parameter(reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size))
        elif attention_matrix == 'O':
            print('Updating weight O')
            model.transformer.h[lnum].attn.out_proj.weight = torch.nn.Parameter(reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size).T)
        
        print("=" * 50)   
        
        return model
        
    @staticmethod
    def get_QKVO_tensor(model, lnum):       
        stacked_tensor = []
        stacked_tensor.append(model.transformer.h[lnum].attn.k_proj.weight.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim))
        stacked_tensor.append(model.transformer.h[lnum].attn.q_proj.weight.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim))
        stacked_tensor.append(model.transformer.h[lnum].attn.v_proj.weight.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim))
        stacked_tensor.append(model.transformer.h[lnum].attn.out_proj.weight.T.view(GPTJLaser.hidden_size, GPTJLaser.n_heads, GPTJLaser.head_dim))
        return torch.stack(stacked_tensor, dim=3)      

    @staticmethod
    def get_QKVO_edited_model(model, lnum, device, qkvo_rank, stack_rank, head_dim_rank=None, qkvo_intervention='partial_tucker', logger=None, in_place=True):
        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)
            
        QKVO_tensor = GPTJLaser.get_QKVO_tensor(model_edit, lnum)

        tl.set_backend('pytorch')
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            gc.collect()

        tensorly_tensor = tl.tensor(QKVO_tensor, device=device)
        
        # Only decompose d_emb and d_h dimensions
        if qkvo_intervention == 'partial_tucker':
            print("=" * 50)            
            print('Partial Tucker decomposition')
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank*GPTJLaser.n_heads, qkvo_rank, stack_rank], init='svd', tol=1e-5, verbose=True)
            print("=" * 50)
            
        elif qkvo_intervention == 'partial_tucker_v2':
            print("=" * 50)            
            print('Partial Tucker decomposition v2')
            
            assert qkvo_rank*(GPTJLaser.n_heads) <= GPTJLaser.head_dim, f"head dim rank: {GPTJLaser.hidden_size}; qkvo_rank*{GPTJLaser.n_heads}: {qkvo_rank*(GPTJLaser.n_heads)}"
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank, qkvo_rank*(GPTJLaser.n_heads), stack_rank], init='svd', tol=1e-5, verbose=True)
            print("=" * 50)
            
        elif qkvo_intervention == 'partial_tucker_v3':
            print("=" * 50)            
            print('Partial Tucker decomposition v3')
            
            assert qkvo_rank*(GPTJLaser.n_heads // 2) <= GPTJLaser.head_dim, f"head dim rank: {GPTJLaser.hidden_size}; qkvo_rank*{GPTJLaser.n_heads // 2}: {GPTJLaser.n_heads // 2}"
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank, qkvo_rank*{GPTJLaser.n_heads // 2}, stack_rank], init='svd', tol=1e-5, verbose=True)            
            print("=" * 50)
            
        elif qkvo_intervention == 'partial_tucker_v4':
            print("=" * 50)            
            print('Partial Tucker decomposition v4')
            
            assert qkvo_rank*(GPTJLaser.n_heads * 2) < GPTJLaser.hidden_size, f"hidden dim rank: {GPTJLaser.hidden_size}; qkvo_rank*{GPTJLaser.n_heads * 2}: {qkvo_rank*(GPTJLaser.n_heads * 2)}"
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank*(GPTJLaser.n_heads * 2), qkvo_rank, stack_rank], init='svd', tol=1e-5, verbose=True)            
            print("=" * 50)

        elif qkvo_intervention == 'partial_tucker_v5':
            # shape=[hidden_size, n_heads, head_dim, 4]
            print("=" * 50)            
            print('Partial Tucker decomposition v5')
            assert qkvo_rank <= GPTJLaser.hidden_size, f'qkvo_rank={qkvo_rank}, hidden dim = {GPTJLaser.hidden_size}'
            assert head_dim_rank <= GPTJLaser.head_dim, f'head_dim_rank={head_dim_rank}, head dim = {GPTJLaser.head_dim}'
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank, head_dim_rank, stack_rank], init='svd', tol=1e-5, verbose=True)
            print("=" * 50)

        reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1], factors[2]], modes=[0, 2, 3])
        reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
        print(f'Reconstruction error: {reconstruction_error}')
        
        model_edit.transformer.h[lnum].attn.k_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,0].reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size))
        model_edit.transformer.h[lnum].attn.q_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,1].reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size))
        model_edit.transformer.h[lnum].attn.v_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,2].reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size))
        model_edit.transformer.h[lnum].attn.out_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,3].reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size).T)
        return model_edit
    

        
        
    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            print(f"Not intervening at all")
            return model_edit           
            
        converted_names = GPTJLaser.convert_name(lname)
        num_update = 0

        for name, param in model.named_parameters():

            modify_flag = GPTJLaser._modify_layer(name=name,
                                                  lnum_to_modify=lnum,
                                                  lname_to_modify=lname,
                                                  converted_names=converted_names)

            if modify_flag:
                if logger is not None:
                    logger.log(f"Updating Layer: {name}")
                print(f"Updating Layer: {name}")
            else:
                continue

            if intervention == 'dropout':
                # For the sparsity analysis
                mat_analysis = param.detach().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)

                mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
                mat_analysis = torch.from_numpy(mat_analysis)

            elif intervention == 'rank-reduction':
                # Do rank reduction
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1)
                
            elif intervention == 'zero':
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)
            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            GPTJLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters GPTJ: {lnum}, {lname}"

        if logger is not None:
            logger.log(f"Total number of parameters updated is {num_update}")

        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
