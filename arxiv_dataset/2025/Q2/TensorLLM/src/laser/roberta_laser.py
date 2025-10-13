import torch

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune
import tensorly as tl
import gc
from tensorly.decomposition import parafac, tucker, tensor_train, partial_tucker


class RobertaLaser(AbstractLaser):
    n_heads = 12
    hidden_size = 768
    head_dim = hidden_size // n_heads
        
    def __init__(self):
        pass

    @staticmethod
    def convert_name(name):

        if name == "k_proj":
            converted_name = "attention.self.key.weight"
        elif name == "q_proj":
            converted_name = "attention.self.query.weight"
        elif name == "v_proj":
            converted_name = "attention.self.value.weight"
        elif name == "out_proj":
            converted_name = "attention.output.dense.weight"
        elif name == "fc_in":
            converted_name = "intermediate.dense.weight"
        elif name == "fc_out":
            converted_name = "output.dense.weight"
        elif name == "mlp":
            converted_name = ["intermediate.dense.weight", "output.dense.weight"]
        elif name == "attn":
            converted_name = ["attention.self.key.weight", "attention.self.query.weight", "attention.self.value.weight", "attention.output.dense.weight"]
        elif name == "all":
            converted_name = ["intermediate.dense.weight", "output.dense.weight", "attention.self.key.weight", 
                              "attention.self.query.weight", "attention.self.value.weight", "attention.output.dense.weight"]
        elif name == "None":
            converted_name = "None"
        else:
            raise AssertionError(f"Unhandled name {name}")

        return converted_name
    
    @staticmethod
    def _modify_layer(name, converted_names, lnum_to_modify):
        # Check if layer type needs to be modified.
        #      'all', 'mlp', 'attn', 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out'
        # If all, then modify all
        # If mlp, then only MLP
        # If attn, then only attn
        # Otherwise, update a given layer type
        if lnum_to_modify != 12 and not name.startswith(f"roberta.encoder.layer.{lnum_to_modify}."):
            return False

        if type(converted_names) == list:
            modify_flag = any([name.endswith(f"{converted_name}") for converted_name in converted_names])
        elif type(converted_names) == str:
            modify_flag = name.endswith(f"{converted_names}")
        else:
            raise AssertionError(f"Type should be list or str. Found {type(converted_names)}.")

        return modify_flag

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            print(f"Not intervening at all")
            return model_edit

        ''' 
            For a given layer, we can modify the following type individually or all at onces
            roberta.encoder.layer.1.attention.self.query.weight
            roberta.encoder.layer.1.attention.self.key.weight
            roberta.encoder.layer.1.attention.self.value.weight
            roberta.encoder.layer.1.attention.output.dense.weight
            roberta.encoder.layer.1.intermediate.dense.weight
            roberta.encoder.layer.1.output.dense.weight
        '''
        # 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None'
        converted_names = RobertaLaser.convert_name(lname)
        num_update = 0
        
        for name, param in model.named_parameters():
            modify_flag = RobertaLaser._modify_layer(name=name, converted_names=converted_names, lnum_to_modify=lnum)
            
            if not modify_flag:
                continue
            
            if lnum != 12 and not name.startswith(f"roberta.encoder.layer.{lnum}."):
                continue

            # if lname != "None" and not name.startswith(f"roberta.encoder.layer.{lnum}.{converted_name}"):
            #     continue

            if logger is not None:
                logger.log(f"Updating Layer: {name}")
            # print(f"Updating Layer: {name}")

            if intervention == 'dropout':
                mat_analysis = param.detach().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)
                
                mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
                mat_analysis = torch.from_numpy(mat_analysis)

            elif intervention == 'rank-reduction':
                # Do rank reduction
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1, niter=20)

            elif intervention == 'zero':
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)

            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            RobertaLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters GPTJ: {lnum}, {lname}"

        return model_edit
    
    @staticmethod
    def get_QKVO_tensor(model, lnum):
        
        
        stacked_tensor = []
        stacked_tensor.append(model.roberta.encoder.layer[lnum].attention.self.key.weight.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim))
        stacked_tensor.append(model.roberta.encoder.layer[lnum].attention.self.query.weight.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim))
        stacked_tensor.append(model.roberta.encoder.layer[lnum].attention.self.value.weight.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim))
        stacked_tensor.append(model.roberta.encoder.layer[lnum].attention.output.dense.weight.T.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim))
        return torch.stack(stacked_tensor, dim=3) # shape=[hidden_size, n_heads, head_dim, 4]
    
    @staticmethod
    def get_3D_Tucker_edited_model(model, lnum, device, qkvo_rank, attention_matrix, in_place=True):
        if attention_matrix == 'Q':
            print('Extracting weight Q')
            tensor = model.roberta.encoder.layer[lnum].attention.self.query.weight.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        elif attention_matrix == 'K':
            print('Extracting weight K')
            tensor = model.roberta.encoder.layer[lnum].attention.self.key.weight.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        elif attention_matrix == 'V':
            print('Extracting weight V')
            tensor = model.roberta.encoder.layer[lnum].attention.self.value.weight.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        elif attention_matrix == 'O':
            print('Extracting weight O')
            tensor = model.roberta.encoder.layer[lnum].attention.output.dense.weight.T.view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        
        tl.set_backend('pytorch')
        # shape = [hidden_size, n_heads, head_dim]
        tensorly_tensor = tl.tensor(tensor, device=device)
        
        print("=" * 50)            
        print('Partial Tucker decomposition')

        assert qkvo_rank <= RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank}"
        (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2], rank=[qkvo_rank*RobertaLaser.n_heads, qkvo_rank], init='svd', tol=1e-5, verbose=True)
        reconstructed_tensor = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1]], modes=[0, 2])

        reconstruction_error = torch.norm(reconstructed_tensor - tensorly_tensor) / torch.norm(tensorly_tensor)
        print(f'Reconstruction error: {reconstruction_error}')
        
        if attention_matrix == 'Q':
            print('Updating weight Q')
            model.roberta.encoder.layer[lnum].attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size))
        elif attention_matrix == 'K':
            print('Updating weight K')
            model.roberta.encoder.layer[lnum].attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size))
        elif attention_matrix == 'V':
            print('Updating weight V')
            model.roberta.encoder.layer[lnum].attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size))
        elif attention_matrix == 'O':
            print('Updating weight O')
            model.roberta.encoder.layer[lnum].attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).T)
        
        print("=" * 50)   
        
        return model
        
            
    @staticmethod
    def get_QKVO_edited_model(model, lnum, device, qkvo_rank, stack_rank, head_dim_rank=None, new_reshape=False, qkvo_intervention='partial_tucker', logger=None, in_place=True):
        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)
            
        QKVO_tensor = RobertaLaser.get_QKVO_tensor(model_edit, lnum)

        tl.set_backend('pytorch')
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            gc.collect()

        tensorly_tensor = tl.tensor(QKVO_tensor, device=device)
        
        # Only decompose d_emb and d_h dimensions
        if qkvo_intervention == 'partial_tucker':
            print("=" * 50)            
            print('Partial Tucker decomposition')

            # shape=[hidden_size, n_heads, head_dim, 4]
            assert qkvo_rank <= RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank}"
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank*RobertaLaser.n_heads, qkvo_rank, stack_rank], init='svd', tol=1e-5, verbose=True)
            reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1], factors[2]], modes=[0, 2, 3])

            reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
            print(f'Reconstruction error: {reconstruction_error}')
            print("=" * 50)
        elif qkvo_intervention == 'partial_tucker_v2':
            print("=" * 50)            
            print('Partial Tucker decomposition v2')

            # shape=[hidden_size, n_heads, head_dim, 4]
            assert qkvo_rank*RobertaLaser.n_heads < RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank*RobertaLaser.n_heads}"
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank, qkvo_rank*RobertaLaser.n_heads, stack_rank], init='svd', tol=1e-5, verbose=True)
            reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1], factors[2]], modes=[0, 2, 3])

            reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
            print(f'Reconstruction error: {reconstruction_error}')
            print("=" * 50)
        elif qkvo_intervention == 'partial_tucker_v3':
            print("=" * 50)            
            print('Partial Tucker decomposition v3')

            # shape=[hidden_size, n_heads, head_dim, 4]
            assert qkvo_rank*(RobertaLaser.n_heads // 2) <= RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank*(RobertaLaser.n_heads // 2)}"
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank, qkvo_rank*(RobertaLaser.n_heads // 2), stack_rank], init='svd', tol=1e-5, verbose=True)
            reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1], factors[2]], modes=[0, 2, 3])

            reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
            print(f'Reconstruction error: {reconstruction_error}')
            print("=" * 50)
        elif qkvo_intervention == 'partial_tucker_v4':
            print("=" * 50)            
            print('Partial Tucker decomposition v4')

            # shape=[hidden_size, n_heads, head_dim, 4]
            assert qkvo_rank*24 <= 768, f"rank exceeds hidden_dim. head_dim=768, rank={qkvo_rank*24}"
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank*24, qkvo_rank, stack_rank], init='svd', tol=1e-5, verbose=True)
            reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1], factors[2]], modes=[0, 2, 3])

            reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
            print(f'Reconstruction error: {reconstruction_error}')
            print("=" * 50)
        elif qkvo_intervention == 'partial_tucker_v5':
            print("=" * 50)            
            print('Partial Tucker decomposition v5')

            # shape=[hidden_size, n_heads, head_dim, 4]
            assert qkvo_rank <= 768, f'qkvo_rank={qkvo_rank}, hidden dim = 768'
            assert head_dim_rank <= RobertaLaser.head_dim, f'head_dim_rank={head_dim_rank}, head dim = {RobertaLaser.head_dim}'
            (core, factors), rec_errors = partial_tucker(tensorly_tensor, modes=[0, 2, 3], rank=[qkvo_rank, head_dim_rank, stack_rank], init='svd', tol=1e-5, verbose=True)
            reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1], factors[2]], modes=[0, 2, 3])

            reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
            print(f'Reconstruction error: {reconstruction_error}')
            print("=" * 50)
            

        model_edit.roberta.encoder.layer[lnum].attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,0].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size))
        model_edit.roberta.encoder.layer[lnum].attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,1].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size))
        model_edit.roberta.encoder.layer[lnum].attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,2].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size))
        model_edit.roberta.encoder.layer[lnum].attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:,:,:,3].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).T)
        return model_edit