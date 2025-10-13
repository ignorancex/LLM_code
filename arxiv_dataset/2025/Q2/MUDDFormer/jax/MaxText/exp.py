class Llama:
    vocab_size = 32000  # TODO
    mlp_activations = ["silu","linear"]
    enable_dropout = False
    logits_via_embedding = False
    normalization_layer_epsilon = 1.0e-6  # TODO
    decoder_block = "llama2"
    # TODO = flash attention

    steps = 10000000
    log_period = 25 # Flushes Tensorboard

    max_target_length = 2049
    mgate = False # lsp
    attention = 'dot_product'
    # dataset_type = 'pile'
    scan_layers = True
    rope_max_timescale = 10000
    
class LlamaMedium(Llama):
    base_emb_dim = 1024
    base_num_query_heads = 16
    base_num_kv_heads = 16
    base_mlp_dim = 2816
    base_num_decoder_layers = 24
    head_dim = 64

    learning_rate = 3.0e-4
    cosine_learning_rate_final_fraction = 0.1
    warmup_steps_fraction = 0.01
    learning_rate_schedule_steps = 13500

    per_device_batch_size = 8.0  # for ICI_MESH_SHAPE = [1, 32, 1]

class Llama7B(Llama):
    base_emb_dim = 4096
    base_num_query_heads = 32
    base_num_kv_heads = 32
    base_mlp_dim = 11008
    base_num_decoder_layers = 32
    head_dim = 128

class MUDDLlama2Medium(LlamaMedium):
    
    # model params
    base_num_decoder_layers = 24

    dense_conn = True # dense_proj1 and dense_proj2
    dynamic_dense_type = 'qkvm'
    dynamic_dense_act_cls = 'gelu'
    dynamic_dense_fix_last_layer = True
    dynamic_dense_hidden_expand = [1] * (base_num_decoder_layers - 1) + [4] # last layer is 4
    dynamic_dense_hidden_round = True
    scan_layers = False
    ddw_gen_pattern = 'q,k,v,m'
    ddw_gen_chunk_size = None
    mudd_prenorm = True
    mudd_postnorm = True
    mudd_in_layer = True
    dynamic_mlp_dim = True # if true: [round( default_dim* (i/(num_layers-1) +0.5) / 128) * 128 for i in range(num_layers)] 
    # opt
    learning_rate_schedule_steps = 13500
    warmup_steps_fraction = 0.01
    cosine_learning_rate_final_fraction = 0.1
    adam_b1 = 0.9
    adam_b2 = 0.95
    adam_eps = 1.0e-8
    adam_weight_decay = 0.1
    # model save
    checkpoint_period = 500
    keep_period = 1000
    # others
    model_name = 'MUDDLlama2Medium'
    learning_rate = 3e-4
    per_device_batch_size = 32.0 # float, v5p-16, core 8, total batch size = 8 * 32 = 256
    eval_per_device_batch_size = 32.0
    eval_interval = 13500
    normalization_layer_epsilon = 1.0e-6
    train_shuffle_buffer_size = None
    eval_shuffle_buffer_size = None
    eval_loop_num_batches = 162
    iter_file_nums = 2
    dataset_type = 'pile'
    vocab_size = 50432
    enable_checkpointing = True
    record_internal_nn_metrics=0

class Llama2Medium(LlamaMedium):
    
    # model params
    base_num_decoder_layers = 24
    dense_conn = False # dense_proj1 and dense_proj2, mudd开关
    dynamic_dense_type = '' # mudd开关
    scan_layers = False
    # opt
    learning_rate_schedule_steps = 13500
    warmup_steps_fraction = 0.01
    cosine_learning_rate_final_fraction = 0.1
    adam_b1 = 0.9
    adam_b2 = 0.95
    adam_eps = 1.0e-8
    adam_weight_decay = 0.1
    # model save
    checkpoint_period = 500
    keep_period = 1000
    # others
    model_name = 'Llama2Medium'
    learning_rate = 3e-4
    per_device_batch_size = 32.0 # float, v5p-16, core 8, total batch size = 8 * 32 = 256
    eval_per_device_batch_size = 32.0
    eval_interval = 13500
    normalization_layer_epsilon = 1.0e-6
    train_shuffle_buffer_size = None
    eval_shuffle_buffer_size = None
    eval_loop_num_batches = 162
    iter_file_nums = 2
    dataset_type = 'pile'
    vocab_size = 50432
    enable_checkpointing = True
