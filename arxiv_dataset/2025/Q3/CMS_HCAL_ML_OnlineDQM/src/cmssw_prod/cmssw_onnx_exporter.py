
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 17:39:37 2025

@author: Mulugeta W.Asres
@email: muleina2000@gmail.com, mulugetawa@uia.no, mulugeta.asres@cern.ch
"""

import argparse
import torch

from model_modulizer import *
import utilities as util
import onnx
import onnxruntime as ort

from torchsummaryX import summary

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

root_path = os.path.dirname(os.path.dirname(current_path))
result_path = "{}//results".format(root_path)

print(torch.__version__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def replace_max_unpool(model):
    def recursive_func(module):
        i = 0
        for name, child in module.named_children():
            if isinstance(child, torch.nn.MaxUnpool3d):
                print("replacing func", name, child, model.cnn_block.d_layer_spatial_dim[i+1])
                unpool_onnx_layer = torch.nn.Upsample(size=tuple(model.cnn_block.d_layer_spatial_dim[i+1]))
                setattr(module, name, unpool_onnx_layer)
                i = i + 1
            else:
                recursive_func(child)
    recursive_func(model)
    model.cnn_block.upsample_layer = "upsample"
    # model.cnn_block.upsample_layer = "unpool"
    return model
              
def get_dummy_inputs(modelProdObj, outpout_mode: str='vector', input_vars: list=['input_data', 'input_data_exo'], memorysize: int=None, seed=123):
    np.random.seed(seed)

    if hasattr(modelProdObj.model_profile['admodel'], "spatial_dims_full"): 
        spatial_dims = modelProdObj.model_profile['admodel'].spatial_dims_full

    if memorysize is None:
        memorysize = modelProdObj.model_profile['admodel'].memorysize
        
    dummy_input = 1e3*np.ones([memorysize] + list(spatial_dims) + [modelProdObj.model_profile['admodel'].feature_dim])
    dummpy_input_data_exo = 2e3*np.random.random([memorysize, len(modelProdObj.data_kwargs['normalizer_vars'])])

    print('dummy_input: {}, dummy_input_data_exo: {}'.format(dummy_input.shape, dummpy_input_data_exo.shape))

    dummy_input = {'input_data': dummy_input, 'input_data_exo': dummpy_input_data_exo}
    dummy_input = {k:v for k,v in dummy_input.items() if k in input_vars}

    if outpout_mode == 'list-default':
        return tuple(value.tolist() if isinstance(value, np.ndarray) else value for value in dummy_input.values())
    else:
        raise NotImplementedError
        
def main(**kwargs):
    model_suffix = kwargs.get("model_suffix", None)
    prod_dir = kwargs.get("prod_dir", "prod_model")
    mode = kwargs.get("mode", "e_s")
    
    if mode == "e_s":

        #############################################################################################    
        # load model
        model_dirpath_prod = rf"{result_path}/{prod_dir}"
        modelProdObj = load_prod_model(model_suffix, model_dirpath=model_dirpath_prod)
        
        assert isinstance(modelProdObj, ModelProdNN), "generate modulized model using model_modulized_exporter.py"

        #############################################################################################    
        # Replace max_unpool with custom function
        if hasattr(modelProdObj.model_profile['admodel'], "model"):
            modelProdObj.model_profile['admodel'].model = replace_max_unpool(modelProdObj.model_profile['admodel'].model)
        else:
            modelProdObj.model_profile['admodel'] = replace_max_unpool(modelProdObj.model_profile['admodel'])
            
        print(modelProdObj.model_profile['admodel'])

        #############################################################################################    
        # load dummy input

        memorysize = 1
        input_vars=['input_data', 'input_data_exo']
        dummy_inputs = get_dummy_inputs(modelProdObj, memorysize=memorysize, outpout_mode='list-default', input_vars=input_vars)

        batch_dim = 1
        if hasattr(modelProdObj.model_profile['admodel'], "use_spatial_split"): # new models
            if modelProdObj.model_profile['admodel'].use_spatial_split:
                batch_dim = 2 # for splitted H and P maps

        print("batch_dim: ", batch_dim)

        #############################################################################################    
        print("model mode stateful...")
        if hasattr(modelProdObj.model_profile['admodel'], "model"):
            e_rnn_hidden, d_rnn_hidden = modelProdObj.model_profile['admodel'].model.rnn_block.get_ae_rnn_states_starter(batch_dim=batch_dim)
        else:
            e_rnn_hidden, d_rnn_hidden = modelProdObj.model_profile['admodel'].rnn_block.get_ae_rnn_states_starter(batch_dim=batch_dim)

        # serialize
        rnn_hidden_dict = {}
        for l, x_layer in enumerate(e_rnn_hidden):
            for s, x in enumerate(x_layer):
                rnn_hidden_dict["e_rnn_hidden__layer_{}_state_{}".format(l, s)] = x.detach().numpy().tolist()

        for l, x_layer in enumerate(d_rnn_hidden):
            for s, x in enumerate(x_layer):
                rnn_hidden_dict["d_rnn_hidden__layer_{}_state_{}".format(l, s)] = x.detach().numpy().tolist()

        eval_setting_dict = {"anomaly_std_th": 10}
        eval_setting_dict.update(rnn_hidden_dict)
        
        for x in dummy_inputs:
            print(len(x))
        
        eval_comapre_output_idx = 1 #pred_data
        dummy_inputs = list(dummy_inputs)
        dummy_inputs_names = input_vars[:]
        for k, v in eval_setting_dict.items():                
            dummy_inputs_names.append(k)
            dummy_inputs.append(v)

        dummy_inputs = tuple([torch.tensor(x, dtype=torch.float32) for x in dummy_inputs])

        print("dummy_inputs shapes...")
        for dummy_input in dummy_inputs:
            print(dummy_input.shape)
            
        print(len(dummy_inputs))
        dummy_inputs_org = copy.deepcopy(dummy_inputs[:])

        modelProdObj.admodel = modelProdObj.model_profile["admodel"] 
        modelProdObj.input_scaler = modelProdObj.model_profile['admodel'].train_data_config["input_scaler"]
        modelProdObj.admodel = modelProdObj.admodel.to(device)

        summary(modelProdObj.admodel, dummy_inputs[0].unsqueeze(0).to(device))


        #############################################################################################    
        # direct py inference
        print('#'*60)
        print("direct inference...")
        modelProdObj.eval()

        with torch.no_grad():
            datainput_dict = {k: v.detach().clone() for k, v in zip(
                dummy_inputs_names, dummy_inputs) if k not in eval_setting_dict.keys()}

            dataObj = modelProdObj.data_pipeline(copy.deepcopy(datainput_dict), isonnx=False)
            result_direct = modelProdObj.pypredict(dataObj, copy.deepcopy(eval_setting_dict))
            result_names = list(result_direct.keys())
            print(result_names)
            print(type(result_names))
            for output_var_name, r in zip(result_names, list(result_direct.values())):
                print(f"{output_var_name}:", r.shape)

            result_direct_single = list(result_direct.values())[eval_comapre_output_idx].detach().numpy()

            print(result_direct_single.dtype, result_direct_single.shape)
            print(f"result_direct [{eval_comapre_output_idx}]: ", result_direct_single.round(4).sum()) 

        
        #############################################################################################    
        print('#'*60)
        print("onnx generation...")  

        onnx_model_full_filepath = model_dirpath_prod + model_suffix + "_stateful.onnx"

        print(onnx_model_full_filepath)

        dummy_inputs = copy.deepcopy(dummy_inputs_org[:])  
        dynamic_axes = {var_name: {0: 'batch_size'} for i, var_name in enumerate(
            dummy_inputs_names) if var_name not in eval_setting_dict.keys()}
        dynamic_axes.update({var_name: {0: 'batch_size'} for i, var_name in enumerate(result_names)})
        util.print_dict(dynamic_axes)

        modelProdObj.admodel = modelProdObj.admodel.to(device)
        modelProdObj.eval()
        print(len(dummy_inputs))
        with torch.no_grad():
            torch.onnx.export(modelProdObj, dummy_inputs, onnx_model_full_filepath,
                            opset_version=15,
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            export_params=True,        # store the trained parameter weights inside the model file
                            verbose=False,
                            input_names=dummy_inputs_names, output_names=result_names,
                            dynamic_axes=dynamic_axes
                            )

            print('loading model...')
            onnx_model = onnx.load(onnx_model_full_filepath)
            onnx.checker.check_model(onnx_model)

        print('#'*60)


        #############################################################################################    
        print("onnx inference checking...")

        print(onnx_model_full_filepath)
        dummy_inputs = copy.deepcopy(dummy_inputs_org[:]) 
        dummy_inputs = tuple([x.detach().clone().numpy() for x in dummy_inputs])

        print(len(dummy_inputs))
        print(dummy_inputs[0].sum(), type(dummy_inputs[0]))

        # load onnx model for inference
        session = ort.InferenceSession(onnx_model_full_filepath)

        input_names = [session.get_inputs()[i].name for i in range(len(session.get_inputs()))]
        print('input_names: ', input_names)

        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        print('output_names: ', output_names)

        result_onnx = session.run(output_names, {
                                input_name: dummy_input for input_name, dummy_input in zip(dummy_inputs_names, dummy_inputs) if input_name in input_names})
        print(len(result_onnx))
        for output_var_name, r in zip(output_names, result_onnx):
            print(f"{output_var_name}: ", r.shape)

        # comparing value at specific eval_comapre_output_idx 
        print(f"result_direct[{eval_comapre_output_idx}]: ", result_direct_single.round(4).sum(), type(result_direct_single))
        print(f"result_onnx[{eval_comapre_output_idx}]: ", result_onnx[eval_comapre_output_idx].round(4).sum(), type(result_onnx[eval_comapre_output_idx]))

        # compare all the results with reference outputs up to 4 decimal places
        for ref_o, o in zip(result_direct.values(), result_onnx):
            np.testing.assert_almost_equal(ref_o.detach().numpy(), o, 4)

        print('ONNX Runtime outputs are similar to reference outputs!')

    else:
        raise "Only rnn stateful onnx conversion mode is available in this exporter."
    
if __name__ == '__main__':
    """Main entry function."""

    parser = argparse.ArgumentParser(description="model exporter source codes")

    parser.add_argument('-pd', '--prod_dir', type=str, default="prod_models",
                        help='production model store directory')
    parser.add_argument('-ms', '--model_suffix', type=str,
                        help='model name tag')
    parser.add_argument('-mode', '--mode', type=str,
                        help='mode: (e)xport or (t)est', default='e')

    args = vars(parser.parse_args())
    print(args)
    main(**args)

# onnx_exporter
# stateful, for CMS_HCAL_ML_OnlineDQM version
# python cmssw_onnx_exporter.py -mode e_s -pd HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB/model/he_m_5_is_minmax_ns_1/prod_models/DQM_AD_ONLINE_m5 -ms GraphSTAD_RIN_DC__CNNRNNAE_MultiDim_SPATIAL_HE_v02_07_2025_13h49




