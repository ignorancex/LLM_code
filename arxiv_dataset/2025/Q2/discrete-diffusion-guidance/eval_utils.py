import os

import torch
import transformers
from tqdm import tqdm

import diffusion


def compute_ppl(
    pretrained_model,
    val_ds
):
  ppl_metrics = diffusion.Perplexity().to('cuda')
  pbar = tqdm(val_ds, desc='PPL')
  for batch in pbar:
    input_ids = batch['input_ids'].to('cuda')
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask'].to('cuda')
    else:
      attention_mask = None
    losses = pretrained_model._loss(input_ids, attention_mask)
    ppl_metrics.update(losses.nlls, losses.token_mask)
    pbar.set_postfix({'ppl': ppl_metrics.compute().item()})
  return ppl_metrics.compute().item()


def compute_generative_ppl(
    sentences,
    eval_model_name_or_path,
    gen_ppl_eval_batch_size=8,
    max_length=128):
  gen_ppl_metric = diffusion.Perplexity().to('cuda')
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
    eval_model_name_or_path)
  if eval_model_tokenizer.pad_token is None:
    eval_model_tokenizer.pad_token = \
      eval_model_tokenizer.eos_token
    eval_model_tokenizer.pad_token_id = \
      eval_model_tokenizer.eos_token_id
  eval_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name_or_path).eval()
  if max_length is None:
    max_length = max_length
  eval_model = eval_model.to('cuda')
  # Re-tokenize using eval model's tokenizer
  tokenizer_kwargs = {
    'return_tensors': 'pt',
    'return_token_type_ids': False,
    'return_attention_mask': True,
    'truncation': True,
    'padding': True,
    'max_length': max_length,
  }
  eval_context_size = 1024
  samples = eval_model_tokenizer(
    sentences, **tokenizer_kwargs)
  attn_mask = samples['attention_mask']
  samples = samples['input_ids']
  attn_mask = attn_mask.to('cuda')
  samples = samples.to('cuda')
  num_batches = samples.shape[0] // gen_ppl_eval_batch_size
  for i in tqdm(range(num_batches),
                desc='Gen. PPL', leave=False):
    _samples = torch.split(
      samples[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    _attn_mask = torch.split(
      attn_mask[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
      logits = eval_model(
        sample_chunk, attention_mask=attn_mask_chunk)[0]
      logits = logits.transpose(-1, -2)

      nlls = torch.nn.functional.cross_entropy(
        logits[..., :-1],
        sample_chunk[..., 1:],
        reduction='none')
      # first_eos = (sample_chunk == eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
      # token_mask = (sample_chunk != eval_model_tokenizer.eos_token_id)
      # gen_ppl_metric.update(
      #   nlls, first_eos[..., 1:] + token_mask[..., 1:])
      gen_ppl_metric.update(
        nlls, attn_mask_chunk[..., 1:])
  return gen_ppl_metric.compute().item()
