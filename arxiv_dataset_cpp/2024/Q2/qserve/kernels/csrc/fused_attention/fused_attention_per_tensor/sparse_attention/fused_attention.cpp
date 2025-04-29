// Adapted from NVIDIA/FasterTransformer and FlashAttention
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
// @article{yang2025lserve,
//   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
//   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
//   year={2025}
// }

#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include <c10/cuda/CUDAGuard.h>

#include "../../common/input_metadata_helper.h"
#include "../../common/kvCacheUtils.h"
#include "../per_tensor_common/update_kv_cache.h"
#include "fused_attention.h"
#include "decoderMaskedMultiheadAttention.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define DISPATCH_FLOAT_AND_HALF_AND_BF16(TYPE, NAME, ...)                  \
  if (TYPE == at::ScalarType::Half) {                                      \
    using scalar_t = at::Half;                                             \
    __VA_ARGS__();                                                         \
  } else {                                                                 \
    AT_ERROR(#NAME, " not implemented for type '", toString(TYPE), "'"); \
  }

// Moved from fused_attention.cpp
template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<at::Half> {
    using Type = uint16_t;
};

template<>
struct SATypeConverter<at::BFloat16> {
    using Type = __nv_bfloat16;
};

template <typename T>
void set_params(Masked_multihead_attention_params<T> &params,
                const size_t batch_size,
                const size_t nheads,
                const size_t nheads_kv,
                const size_t memory_max_seqlen,
                const size_t headdim,
                const int timestep,
                const int rotary_embedding_dim,
                const float rotary_base,
                const float rotary_embedding_scale,
                const RotaryScalingType rotary_embedding_scale_type,
                const bool neox_rotary_style,
                const int qkv_batch_stride,
                const int tokens_per_block,
                T *q_ptr,
                T *k_ptr,
                T *v_ptr,
                float *kv_scale_quant_orig,
                float *kv_scale_orig_quant,
                // half* k_scale_orig_quant,
                // half* v_scale_orig_quant,
                // T *k_cache_ptr,
                // T *v_cache_ptr,
                int *length_per_sample,
                bool int4_kv_cache,
                bool kv_cache_with_zeros,
                float *alibi_slopes_ptr,
                T *out_ptr,
                const int* retrieval_head_flags_ptr,
                const int* head_rank_table_ptr,
                const int* dynamic_sparse_page_idxes_ptr,
                const int num_retrieval_kv_heads,
                const int num_streaming_kv_heads,
                const int streaming_sink_token_num,
                const int streaming_local_token_num,
                const int num_dynamic_sparse_pages,
                const bool do_dynamic_sparse,
                // add by JXGuo: for multiblock
                int smem_preload_switch,
                int multiblock_switch,
                bool multi_block_mode,
                int max_seq_len_tile,
                T *partial_out,
                float *partial_sum,
                float *partial_max,
                int *block_counter
                ) {
    // Reset the parameters
    // memset(&params, 0, sizeof(params));
    params.q = q_ptr;
    params.k = k_ptr;
    params.v = v_ptr;
    params.q_bias = nullptr;
    params.k_bias = nullptr;
    params.v_bias = nullptr;
    // params.k_cache = k_cache_ptr;
    // params.v_cache = v_cache_ptr;
    // params.linear_bias_slopes = alibi_slopes_ptr;
    params.out = out_ptr;
    params.cache_indir = nullptr;
    // Haotian: be very careful about qkv_batch_stride.
    // k and v are not contiguous!!
    params.stride = qkv_batch_stride;
    params.batch_size = batch_size;
    params.beam_width = 1;
    params.memory_max_len = memory_max_seqlen;
    params.num_heads = nheads;
    params.num_kv_heads = nheads_kv;
    params.hidden_size_per_head = headdim;
    params.rotary_embedding_dim = rotary_embedding_dim;
    params.rotary_embedding_base = rotary_base;
    params.rotary_embedding_scale = rotary_embedding_scale;
    params.rotary_embedding_scale_type = rotary_embedding_scale_type;
    // params.k_scale_quant_orig = k_scale_quant_orig;
    // params.v_scale_quant_orig = v_scale_quant_orig;
    // params.k_scale_orig_quant = k_scale_orig_quant;
    // params.v_scale_orig_quant = v_scale_orig_quant;
    // params.neox_rotary_style = neox_rotary_style;
    params.tokens_per_block = tokens_per_block;
    params.timestep = timestep;
    params.inv_sqrt_dh = 1.f / sqrt(float(headdim));
    // params.total_padding_tokens = nullptr;
    // params.masked_tokens = nullptr;
    // params.prefix_prompt_lengths = nullptr;
    // params.max_prefix_prompt_length = 0;
    params.relative_attention_bias = nullptr;
    params.relative_attention_bias_stride = 0;
    // params.cross_attention_out = nullptr;
    params.max_decoder_seq_len = 0;
    // params.is_return_cross_attentions = false;
    params.finished = nullptr;
    params.memory_length_per_sample = nullptr;
    params.length_per_sample = length_per_sample;
    params.kv_scale_quant_orig = kv_scale_quant_orig;
    params.kv_scale_orig_quant = kv_scale_orig_quant;
    params.int4_kv_cache = int4_kv_cache;
    params.kv_cache_with_zeros = kv_cache_with_zeros;
    params.retrieval_head_flags_ptr = retrieval_head_flags_ptr;
    params.head_rank_table_ptr = head_rank_table_ptr;
    params.dynamic_sparse_page_idxes_ptr = dynamic_sparse_page_idxes_ptr;
    params.num_retrieval_kv_heads = num_retrieval_kv_heads;
    params.num_streaming_kv_heads = num_streaming_kv_heads;
    params.streaming_sink_token_num = streaming_sink_token_num;
    params.streaming_local_token_num = streaming_local_token_num;
    params.num_dynamic_sparse_pages = num_dynamic_sparse_pages;
    params.do_dynamic_sparse = do_dynamic_sparse;
    // std::cout << params.batch_size << " " << params.memory_max_len << " " << params.num_heads << " " << params.hidden_size_per_head << " " << params.timestep << std::endl;
    // add by JXGuo: for multiblock
    params.smem_preload_switch = smem_preload_switch;
    params.multiblock_switch = multiblock_switch;
    params.multi_block_mode = multi_block_mode;
    params.max_seq_len_tile = max_seq_len_tile;
    params.partial_out = partial_out;
    params.partial_sum = partial_sum;
    params.partial_max = partial_max;
    params.block_counter = block_counter;
}


// output = fused_attention.single_query_attention(
//   query,
//   key,
//   value,
//   # block_tables
//   input_metadata.block_tables,
//   # params.lengths_per_sample
//   input_metadata.context_lens,
//   # params.memory_max_len,
//   input_metadata.max_context_len,
//   # block_size,
//   block_size,
//   # params.linear_bias_slopes
//   alibi_slopes,
//   # RoPE parameters below: we do not apply RoPE in this kernel.
//   0,
//   10000,
//   True,
// )

torch::Tensor single_query_attention(const torch::Tensor q,
                                     const torch::Tensor k,
                                     const torch::Tensor v,
                                     c10::optional<const torch::Tensor> kv_scale_quant_orig_,
                                     c10::optional<const torch::Tensor> kv_scale_orig_quant_,
                                     c10::optional<torch::Tensor> _retrieval_kv_pointers, // B x 2 x M
                                     c10::optional<torch::Tensor> _streaming_kv_pointers, // B x 2 x M
                                     torch::Tensor retrieval_head_flags, // H
                                     torch::Tensor head_rank_table, // H
                                     c10::optional<torch::Tensor> dynamic_sparse_page_idxes_,      // B x N_head x dynamic_sparse_page_num
                                     c10::optional<const torch::Tensor> length_per_sample_,
                                     c10::optional<const torch::Tensor> alibi_slopes_,
                                    //  c10::optional<const torch::Tensor> k_scale_orig_quant,
                                    //  c10::optional<const torch::Tensor> v_scale_orig_quant,
                                     int memory_max_seqlen,
                                     int tokens_per_block,
                                     const int size_per_retrieval_token,   // default = hidden_size * sizeof(dtype)
                                     const int size_per_streaming_token,   // default = hidden_size * sizeof(dtype)
                                     const int sink_token_num, const int local_token_num,
                                     const int sink_block_num, const int local_block_num,
                                     const int num_retrieval_kv_heads,
                                     const int num_streaming_kv_heads,
                                     const int timestep,
                                     const int rotary_embedding_dim,
                                     const float rotary_base,
                                     const float rotary_embedding_scale,
                                     // neox_rotary_style = not interleaved
                                     const bool neox_rotary_style,
                                     const bool int4_kv_cache,
                                     const bool kv_cache_with_zeros,
                                     const int tokens_per_sub_chunk,
                                     const int hidden_dim_per_retrieval_token,
                                    //  const int dynamic_sparse_n_indicator_per_sub_chunk,
                                     const int multiblock_switch) {  
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); //CHECK_DEVICE(kv_pointers);
    int batch_size = q.size(0);
    int nheads = q.size(1);
    int nheads_kv = k.size(1);
    int headdim = k.size(-1);

    // int max_blocks = kv_pointers.size(-1)
    // CHECK_SHAPE(q, batch_size, nheads, headdim);
    // CHECK_SHAPE(k, batch_size, nheads_kv, headdim);
    // CHECK_SHAPE(v, batch_size, nheads_kv, headdim);
    // CHECK_SHAPE(kv_pointers, batch_size, 2, max_blocks);
    // TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == headdim);
    TORCH_CHECK(k.stride(2) == 1 && k.stride(1) == headdim);
    TORCH_CHECK(v.stride(2) == 1 && v.stride(1) == headdim);
    // TORCH_CHECK(q.stride(0) == k.stride(0) && q.stride(0) == v.stride(0));
    // CHECK_CONTIGUOUS(kv_pointers);
    // CHECK_CONTIGUOUS(k);
    if (length_per_sample_.has_value()) {
        auto length_per_sample = length_per_sample_.value();
        CHECK_DEVICE(length_per_sample);
        CHECK_SHAPE(length_per_sample, batch_size);
        CHECK_CONTIGUOUS(length_per_sample);
        TORCH_CHECK(length_per_sample.dtype() == torch::kInt32);
    }
    if (kv_scale_quant_orig_.has_value()){
      auto kv_scale_quant_orig = kv_scale_quant_orig_.value();
      CHECK_DEVICE(kv_scale_quant_orig);
      CHECK_CONTIGUOUS(kv_scale_quant_orig);
      TORCH_CHECK(kv_scale_quant_orig.dtype() == torch::kFloat32);
    }
    if (kv_scale_orig_quant_.has_value()){
      auto kv_scale_orig_quant = kv_scale_orig_quant_.value();
      CHECK_DEVICE(kv_scale_orig_quant);
      CHECK_CONTIGUOUS(kv_scale_orig_quant);
      TORCH_CHECK(kv_scale_orig_quant.dtype() == torch::kFloat32);
    }
    if (alibi_slopes_.has_value()) {
      auto alibi_slopes = alibi_slopes_.value();
      CHECK_DEVICE(alibi_slopes);
      CHECK_SHAPE(alibi_slopes, nheads);
      CHECK_CONTIGUOUS(alibi_slopes); 
      TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32);
    }
    
    auto num_dynamic_sparse_pages = dynamic_sparse_page_idxes_.has_value() ? dynamic_sparse_page_idxes_.value().size(-1) : 0;
    bool do_dynamic_sparse = dynamic_sparse_page_idxes_.has_value() && num_dynamic_sparse_pages > 0;
    if (dynamic_sparse_page_idxes_.has_value()) {
      auto dynamic_sparse_page_idxes = dynamic_sparse_page_idxes_.value();
      CHECK_DEVICE(dynamic_sparse_page_idxes);
      CHECK_SHAPE(dynamic_sparse_page_idxes, batch_size, nheads, num_dynamic_sparse_pages);
      CHECK_CONTIGUOUS(dynamic_sparse_page_idxes);
      TORCH_CHECK(dynamic_sparse_page_idxes.dtype() == torch::kInt32);
    }

    int retrieval_max_blocks_per_seq = 0;
    int streaming_max_blocks_per_seq = 0;
    if (_retrieval_kv_pointers.has_value()) {
      auto retrieval_kv_pointers = _retrieval_kv_pointers.value();
      retrieval_max_blocks_per_seq = retrieval_kv_pointers.size(-1);
      CHECK_DEVICE(retrieval_kv_pointers);
      CHECK_CONTIGUOUS(retrieval_kv_pointers);
    }

    if (_streaming_kv_pointers.has_value()) {
      auto streaming_kv_pointers = _streaming_kv_pointers.value();
      streaming_max_blocks_per_seq = streaming_kv_pointers.size(-1);
      CHECK_DEVICE(streaming_kv_pointers);
      CHECK_CONTIGUOUS(streaming_kv_pointers);
    }

    KVBlockArray<false> retrieval_kv_buffer(batch_size, retrieval_max_blocks_per_seq, tokens_per_block, size_per_retrieval_token, 0, 0, 0, 0, tokens_per_sub_chunk, hidden_dim_per_retrieval_token);
    KVBlockArray<true> streaming_kv_buffer(batch_size, streaming_max_blocks_per_seq, tokens_per_block, size_per_streaming_token, sink_token_num, local_token_num, sink_block_num, local_block_num, 0, 0);
    retrieval_kv_buffer.data = _retrieval_kv_pointers.has_value() ? _retrieval_kv_pointers.value().data_ptr<int64_t>() : nullptr;
    streaming_kv_buffer.data = _streaming_kv_pointers.has_value() ? _streaming_kv_pointers.value().data_ptr<int64_t>() : nullptr;
     
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    torch::Tensor out = torch::empty_like(q); 

    const int smem_preload_switch = 2048;
    bool multi_block_mode = false;
    int max_seq_len_tile = 1;
    RotaryScalingType rotary_embedding_scale_type;
    if (rotary_embedding_scale != 1.0f)
      rotary_embedding_scale_type = RotaryScalingType::kLINEAR;
    else
      rotary_embedding_scale_type = RotaryScalingType::kNONE;

    DISPATCH_FLOAT_AND_HALF_AND_BF16(q.scalar_type(), "single_query_attention", [&] {
        using DataType = typename SATypeConverter<scalar_t>::Type;

        DataType* partial_out = nullptr;
        float* partial_sum = nullptr;
        float* partial_max = nullptr;
        int* block_counter = nullptr;

        int _timestep_for_multi_block_switch = timestep;
        if (do_dynamic_sparse)
        {
          _timestep_for_multi_block_switch = (num_dynamic_sparse_pages - 1) * tokens_per_block + (timestep - 1) % tokens_per_block + 1;
        }

        if (_timestep_for_multi_block_switch >= multiblock_switch){
            multi_block_mode = true;
            // int max_shared_mem;
            // cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
            max_seq_len_tile = 100;
            auto partial_out_options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
            at::Tensor _partial_out = torch::zeros({max_seq_len_tile, batch_size, nheads * headdim}, partial_out_options);
            auto partial_sum_options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
            at::Tensor _partial_sum = torch::zeros({batch_size, nheads, max_seq_len_tile}, partial_sum_options);
            at::Tensor _partial_max = torch::zeros({batch_size, nheads, max_seq_len_tile}, partial_sum_options);
            auto block_counter_options = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
            at::Tensor _block_counter = torch::zeros({batch_size, nheads}, block_counter_options);
            partial_out = reinterpret_cast<DataType*>(_partial_out.data_ptr());
            partial_sum = reinterpret_cast<float*>(_partial_sum.data_ptr());
            partial_max = reinterpret_cast<float*>(_partial_max.data_ptr());
            block_counter = reinterpret_cast<int*>(_block_counter.data_ptr());
        }

        Masked_multihead_attention_params<DataType> params;
        
        params.int8_kv_cache = true; 
        set_params(params, batch_size, nheads, nheads_kv, memory_max_seqlen, headdim, 
                   timestep, rotary_embedding_dim, rotary_base, rotary_embedding_scale, rotary_embedding_scale_type,
                   neox_rotary_style, q.stride(0), tokens_per_block,
                   reinterpret_cast<DataType*>(q.data_ptr()),
                   reinterpret_cast<DataType*>(k.data_ptr()),
                   reinterpret_cast<DataType*>(v.data_ptr()),
                   kv_scale_quant_orig_.has_value() 
                       ? kv_scale_quant_orig_.value().data_ptr<float>() : nullptr,
                   kv_scale_orig_quant_.has_value() 
                       ? kv_scale_orig_quant_.value().data_ptr<float>() : nullptr,
                   length_per_sample_.has_value()
                       ? length_per_sample_.value().data_ptr<int>() : nullptr,
                   int4_kv_cache,
                   kv_cache_with_zeros,
                   alibi_slopes_.has_value() 
                       ? alibi_slopes_.value().data_ptr<float>(): nullptr,
                   reinterpret_cast<DataType*>(out.data_ptr()),
                   retrieval_head_flags.data_ptr<int>(),
                   head_rank_table.data_ptr<int>(),
                   dynamic_sparse_page_idxes_.has_value()
                       ? dynamic_sparse_page_idxes_.value().data_ptr<int>() : nullptr,
                   num_retrieval_kv_heads, num_streaming_kv_heads,
                   sink_token_num, local_token_num,
                   num_dynamic_sparse_pages, do_dynamic_sparse,
                   smem_preload_switch,
                   multiblock_switch,
                   multi_block_mode,
                   max_seq_len_tile,
                   partial_out,
                   partial_sum,
                   partial_max,
                   block_counter
                   );
        auto stream = at::cuda::getCurrentCUDAStream();
        masked_multihead_attention(params, retrieval_kv_buffer, streaming_kv_buffer, stream);
    });
    // return torch::zeros_like(q);
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "single_query_attention",
    &single_query_attention,
    "single query attention kernel from trtllm");
  // m.def(
  //   "apply_bias_rope_update_kv_cache",
  //   &apply_bias_rope_update_kv_cache,
  //   "(context stage) add bias, apply rope and update kv cache");
  m.def(
    "compute_padding_offsets",
    &compute_padding_offsets,
    "compute padding offsets");
}
