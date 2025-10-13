import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from trainer.arl import AsyncRLGRPOTrainer
from configs import GRPOTrainingConfig,GRPOScriptArguments
from trl import ModelConfig, TrlParser
from trainer.utils import action_schema_check, action_args_check, GUIRFTDataset,action_type_check,GUIMTRFTDataset,react_check,fsdp2_prepare_model
import torch.distributed as dist

reward_funcs_registry = {
    # "accuracy": iou_reward,
    # "format": format_reward,
    "react": react_check,
    "schema": action_schema_check,
    "type":action_type_check,
    "args":action_args_check,
}

# ----------------------- Main Script -----------------------
def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)


    model_init_kwargs = {}
    model_init_kwargs["attn_implementation"] = model_args.attn_implementation
    model_init_kwargs["torch_dtype"] = torch.bfloat16
    model_init_kwargs["trust_remote_code"] = True
    model_id = model_args.model_name_or_path
    if "minicpm" in model_id.lower():
        if "minicpm-o" in model_id.lower():
            model_init_kwargs["init_tts"] = False
            model_init_kwargs["init_audio"] = False
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
    processing_class = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)
    # processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
    # if processing_class.pad_token_id is None:
    #     processing_class.tokenizer.pad_token_id =  2
    #     processing_class.pad_token_id = 2
    # processing_class.padding_side = "left"
    # processing_class.tokenizer.padding_side = "left"
    device_mesh = None
    if training_args.tensor_parallel_size is not None:
        tp_size = int(training_args.tensor_parallel_size)
        world_size = dist.get_world_size()
        if world_size % tp_size != 0:
            raise ValueError(
                f"world_size {world_size} must be divisible by tensor_parallel_size {tp_size}"
            )
        dp_size = world_size // tp_size

        device_mesh = dist.device_mesh.DeviceMesh(
            "cuda",
            mesh=torch.arange(world_size).reshape((dp_size, tp_size)),
            mesh_dim_names=("dp", "tp"),
        )
        
        tp_mesh = device_mesh["tp"]
        dp_mesh = device_mesh["dp"]
        
        from torch.distributed.tensor.parallel import ColwiseParallel,RowwiseParallel,parallelize_module,SequenceParallel,PrepareModuleInput,PrepareModuleOutput
        from torch.distributed.tensor import  Replicate,Shard

        layer_tp_plan = {
            "llm.model.layers.*.mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),)
            ),
            "llm.model.layers.*.mlp.up_proj": ColwiseParallel(),
            "llm.model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "llm.model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "llm.model.layers.*.self_attn": PrepareModuleInput(
                input_kwarg_layouts={
                    "hidden_states": Shard(1),
                    "attention_mask": None,
                    "position_ids": None,
                    "past_key_value": None,
                    "output_attentions": None,
                    "use_cache": None,
                    "cache_position": None,
                    "position_embeddings": None, 
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                    "attention_mask": None,
                    "position_ids": None,
                    "past_key_value": None,
                    "output_attentions": None,
                    "use_cache": None,
                    "cache_position": None,
                    "position_embeddings": None, 
                },
            ),
            "llm.model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "llm.model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "llm.model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "llm.model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "llm.model.layers.*.input_layernorm": SequenceParallel(),
            "llm.model.layers.*.post_attention_layernorm": SequenceParallel(),
            "llm.model.layers.*.input_layernorm": SequenceParallel(),
            "llm.model.norm": SequenceParallel(),
            "llm.model.layers.0": PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Shard(1),)
            )
            # "llm.model.embed_tokens": 
            # RowwiseParallel(
            #     input_layouts=Replicate(),
            #     output_layouts=Shard(1),
            # ),
            # "llm.lm_head": ColwiseParallel(
            #     input_layouts=Shard(1),
            #     output_layouts=Replicate(),
            #     # use_local_output=False, # TODO: Is this ture for grpo ?
            # )
        }
        
        layer_tp_plan = {
            "llm.model.layers.*.mlp.up_proj": ColwiseParallel(),
            "llm.model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "llm.model.layers.*.mlp.down_proj": RowwiseParallel(),
            "llm.model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "llm.model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "llm.model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "llm.model.layers.*.self_attn.o_proj": RowwiseParallel(),
        }

        model = parallelize_module(
            model,
            tp_mesh,
            layer_tp_plan
        )
        
        # for layer_id, transformer_block in enumerate(model.vpm.encoder.layers):
        #     layer_tp_plan = {
        #         "mlp.fc1": ColwiseParallel(),
        #         "mlp.fc2": RowwiseParallel(),
        #         "self_attn.k_proj": ColwiseParallel(),
        #         "self_attn.q_proj": ColwiseParallel(),
        #         "self_attn.v_proj": ColwiseParallel(),
        #         "self_attn.out_proj": RowwiseParallel()
        #     }
        #     # adjuest attention module to use the local number of heads
        #     attn_layer = transformer_block.self_attn
        #     attn_layer.num_heads = attn_layer.num_heads // tp_mesh.size()
        #     attn_layer.embed_dim = attn_layer.embed_dim // tp_mesh.size()
            
        #     model.vpm.encoder.layers[layer_id]  = parallelize_module(
        #         module=transformer_block,
        #         device_mesh=tp_mesh,
        #         parallelize_plan=layer_tp_plan
        #     )
        
        model = fsdp2_prepare_model(
            model,
            mesh=dp_mesh
        )
    
    global_task_dispatch_addr =  f"tcp://{os.environ.get('MASTER_ADDR')}:{training_args.global_data_dispatch_port}"

    dataset_cls = GUIMTRFTDataset if training_args.hist_length > 1 else GUIRFTDataset
    
    datasets = dataset_cls(
        global_task_dispatch_addr=global_task_dispatch_addr,
        hist_length=training_args.hist_length,
        jsonl_file_path=script_args.dataset_name,
        max_line_res=script_args.max_line_res,)
    
    if script_args.eval_dataset_name is not None:
        eval_set = dataset_cls(
            global_task_dispatch_addr=global_task_dispatch_addr,
            hist_length=training_args.hist_length,
            jsonl_file_path=script_args.eval_dataset_name,
            max_line_res=script_args.max_line_res,)
    else:
        eval_set = None
        
    # Initialize the GRPO trainer
    trainer = AsyncRLGRPOTrainer(
        model=model,
        model_init_kwargs=model_init_kwargs,
        processing_class=processing_class,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=datasets,
        eval_dataset=eval_set,
        device_mesh=device_mesh
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOTrainingConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
