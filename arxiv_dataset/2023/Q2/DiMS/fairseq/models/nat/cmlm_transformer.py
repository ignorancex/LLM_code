# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel
from fairseq.utils import new_arange
import torch
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.torch_imputer import best_alignment
from torch.nn.utils.rnn import pad_sequence


def _block_unmasking(output_scores, output_masks, p, blk_size):
    B, S = output_scores.shape
    blk_size=blk_size if isinstance(blk_size, int) else blk_size[0].item()
    div_pad = torch.full((B, blk_size - S%blk_size), 0, device=output_scores.device)
    output_scores = torch.cat((output_scores, div_pad), dim=-1).reshape(B, -1, blk_size)
    output_masks = torch.cat((output_masks, div_pad), dim=-1).reshape(B, -1, blk_size)
    output_scores[:, :, -1]=0.
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
            (output_masks.sum(-1, keepdim=True).type_as(output_scores) ) * (p if isinstance(p, float) else p.reshape(-1, 1, 1))
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(2, sorted_index, skeptical_mask).reshape(B, -1)[:, :S]


def _block_mask(target_tokens, tgt_dict, blk_size, unk_cnt, force_emits=None):
    unk = tgt_dict.unk()
    pad = tgt_dict.pad()
    B, S = target_tokens.shape
    div_pad = torch.full((B, blk_size.item() - S%blk_size.item()), pad, device=target_tokens.device)
    target_tokens = torch.cat((target_tokens, div_pad), dim=-1).reshape(B, -1, blk_size)
    target_masks = target_tokens.ne(pad)
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_rank = target_score.sort(-1)[1]
    target_cutoff = new_arange(target_rank) < torch.minimum(unk_cnt.unsqueeze(-1).expand(-1, target_tokens.shape[1]), target_masks.sum(-1)).unsqueeze(-1) 
    prev_target_tokens = target_tokens.masked_fill(
        target_cutoff.scatter(2, target_rank, target_cutoff), unk
    )
    prev_target_tokens = prev_target_tokens.reshape(B, -1)[:, :S]
    if force_emits is not None:
        force_emits[prev_target_tokens==unk]=-1
    return prev_target_tokens, force_emits


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def _random_mask(target_tokens, tgt_dict, force_emits=None, full_mask=False):
    pad = tgt_dict.pad()
    bos = tgt_dict.bos()
    eos = tgt_dict.eos()
    unk = tgt_dict.unk()

    target_masks = (
        target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos) 
    )
   
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_length = target_masks.sum(1).float()
    if not full_mask:
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token. 

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < target_length[:, None].long()
    prev_target_tokens = target_tokens.masked_fill(
        target_cutoff.scatter(1, target_rank, target_cutoff), unk
    )
    if force_emits is not None:
        force_emits[prev_target_tokens.eq(unk)]=-1
    return prev_target_tokens, force_emits
    

@register_model("cmlm_transformer")
class CMLMNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()
        self.eos = decoder.dictionary.eos()
        self.bos = decoder.dictionary.bos()
        if getattr(args, 'fine_tune', False):
            print('load checkpoint')
            self.load_state_dict(torch.load(args.fine_tune)['model'])
    
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument('--insertCausalSelfAttn', default=False, action='store_true',
                            help='if set, add an additional unmasked self attention sublayer in the decoder layer; note '
                                 'that this must be used together with the NAT framework, else will cause leaking')
        parser.add_argument('--concatPE', default=False, action='store_true',
                            help='if set, instead of summing embedding vectors to PE, concatenate the halved dimensions')
        parser.add_argument('--layer-prediction-loss-factor', default=0., type=float,
                            help='Deep prediction Loss Factor.')
        parser.add_argument('--valid-per-epoch', default=0, type=int,
                            help='Number of validation done per epoch.')
        parser.add_argument('--align-noise', default=False, action='store_true',
                            help='Rotate the blanks.')
        parser.add_argument('--mask-policy', default='uniform', type=str,
                help='Masking policy: Block or Uniform.')
        parser.add_argument('--ctc-pretrain-step', default=0, type=int,
            help='Number of steps to run ctc before imputer.')
        parser.add_argument('--fine-tune', default=None, type=str,
            help='Checkpoint to finetune.')
        parser.add_argument('--fine-tune-dslp', default=False, action='store_true',
            help='Checkpoint to finetune.')





    def find_best_align(self, src_tokens, tgt_tokens, encoder_out):
        aligned = self.twice_src_masked(src_tokens, tgt_tokens).output_tokens
        with torch.no_grad():
            net_output=self.decoder(
                normalize=True,
                prev_output_tokens=aligned,
                encoder_out=encoder_out,
                inner_states=False
            )

            aligned_mask = aligned.eq(self.unk)
            aligned_length = aligned.ne(self.pad).sum(-1)
            target_lengths = tgt_tokens.ne(self.pad).sum(-1).long()

            force_emits = best_alignment(net_output.permute(1, 0, 2).float(), tgt_tokens, aligned_length, target_lengths, blank=self.blank, zero_infinity=True)
            best_alignments = pad_sequence(list(map(lambda x: torch.tensor(x, device=src_tokens.device), force_emits)), batch_first=True, padding_value=-1)
            force_emits = pad_sequence([torch.tensor(fe) for fe in force_emits], batch_first=True, padding_value=self.pad).cuda()
            best_alignments[(best_alignments%2==0)]=0
            alignement_tokens = torch.gather(tgt_tokens, dim=1, index=torch.div(best_alignments, 2).long()) 
            alignement_tokens[best_alignments==0]=self.blank  
            alignement_tokens[best_alignments==-1]=self.pad
            if getattr(self.args, 'align_noise', False):
                for i in range(len(alignement_tokens)):
                    a = torch.sum(torch.cumsum(alignement_tokens[i].ne(self.blank), dim=-1) == 0)
                    b = torch.sum(torch.cumsum(torch.flip(alignement_tokens[i][:aligned_length[i]].ne(self.blank),dims=(-1,)) , dim=-1) == 0)
                    offset = torch.randint(-a.item(), b.item()+1, (1,)).item()
                    alignement_tokens[i][:aligned_length[i]] = torch.roll(alignement_tokens[i][:aligned_length[i]], offset, dims=(-1,))
                    force_emits[i][:aligned_length[i]] = torch.roll(force_emits[i][:aligned_length[i]], (offset,), dims=(-1,))
        return alignement_tokens, net_output, force_emits

    def add_length_loss(self, return_dict, encoder_out, tgt_tokens):
        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        return_dict["length"]= {
            "out": length_out,
            "tgt": length_tgt,
            "factor": self.decoder.length_loss_factor,
        }

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        
        if getattr(self.args, 'ctc', False):
            best_alignemnt, _, force_emits = self.find_best_align(src_tokens, tgt_tokens, encoder_out)
            prev_output_tokens = best_alignemnt.clone()
            
            if getattr(self.args, 'mask_policy', 'uniform')=='uniform':
                prev_output_tokens, force_emits =_random_mask(prev_output_tokens, self.decoder.dictionary, force_emits=force_emits, full_mask=getattr(self.args, 'noise', 'random_mask')=='full_mask')
            else:
                #blk_size=torch.tensor(8, device=best_alignemnt.device)
                blk_size = 2**torch.randint(low=1, high=5, size=(), device=prev_output_tokens.device)
                #blk_size = torch.randint(low=2, high=9, size=(), device=prev_output_tokens.device) 
                unk_count_per_block = torch.randint(low=1, high=blk_size+1, size=(len(prev_output_tokens),), device=prev_output_tokens.device)
                if getattr(self.args, 'noise', 'random_mask')=='full_mask':
                    unk_count_per_block = torch.ones_like(unk_count_per_block)*blk_size
                prev_output_tokens, force_emits =_block_mask(prev_output_tokens, self.decoder.dictionary, force_emits=force_emits, blk_size=blk_size, unk_cnt=unk_count_per_block)
        # decoding
        word_ins_out, inner_states = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inner_states=True
        )
        embed_states = inner_states['inner_states']
        word_ins_mask = prev_output_tokens.eq(self.unk)

        return_dict= {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask if not getattr(self.args, 'ctc', False) else force_emits, 
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": self.args.layer_prediction_loss_factor/(len(embed_states)-1) if self.args.layer_prediction_loss_factor>0. else 1.
            },
        }
        if self.args.layer_prediction_loss_factor>0:
            self.per_layer_loss(inner_states['inner_logits'], return_dict, tgt_tokens, word_ins_mask, force_emits)
        
        if getattr(self.args, 'ctc', False):
            return return_dict, best_alignemnt
        else:
            self.add_length_loss(return_dict, encoder_out, tgt_tokens)
            return return_dict

    def per_layer_loss(self, inner_logits, return_dict, tgt_tokens, word_ins_mask, force_emits):
        for i in range(len(inner_logits)):
            return_dict[f"word_ins_layer_{i+1}"]={
                "out": inner_logits[i].permute(1, 0, 2),
                "tgt": tgt_tokens ,
                "mask": word_ins_mask if not getattr(self.args, 'ctc', False) else force_emits, 
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": self.args.layer_prediction_loss_factor/(len(inner_logits)+1)
            }

    def initialize_output_tokens(self, encoder_out, src_tokens):
        if getattr(self.args, 'ctc', False):
            return self.twice_src_masked(src_tokens)
        else:
            return super().initialize_output_tokens(encoder_out, src_tokens)

    def twice_src_masked(self, src_tokens, tgt_tokens=None):
        length_tgt = torch.sum(src_tokens.ne(self.pad), -1)*2+30
        length_tgt = torch.sum(src_tokens.ne(self.pad), -1)*2
        if tgt_tokens is not None:
            length_tgt = torch.maximum(length_tgt, torch.sum(tgt_tokens.ne(self.pad), -1))
        max_length = length_tgt.clamp_(min=2).max()
        idx_length = new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).half()
        
        if getattr(self.args, 'fine_tune_dslp', False):
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )


    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder

        output_masks = output_tokens.eq(self.unk)
        if getattr(self.args, 'insertCausalSelfAttn', False) or getattr(self.args, 'ctc_distill', False):
            output_masks = output_tokens.ne(self.pad) & output_tokens.ne(self.bos) & output_tokens.ne(self.eos)

        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)

        

        #probs = F.log_softmax(probs/1., -1)
        #_tokens = probs.argmax(-1)

        # else:
        #     probs = F.log_softmax(probs/1.2, -1)
        #     _tokens = torch.distributions.categorical.Categorical(probs=torch.exp(probs)).sample()
        # _scores = torch.gather(probs, index=_tokens.unsqueeze(-1), dim=-1).squeeze()
       
        if getattr(self.args, 'ctc', False):
            _scores=_scores.half()
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
        # if not all(step==max_step):
            if getattr(self.args, 'mask_policy', 'block')=='uniform':
                skeptical_mask = _skeptical_unmasking(
                    output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
                )
            else:
                skeptical_mask = _block_unmasking(
                    output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step,
                    max_step
                )


            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())
        
            # just for ablation of beam search
            # selects two tokens for least confident ones
            # five_by_five = torch.arange(len(output_scores)//5)*5
            # sorted_index =  _scores[five_by_five].sort(-1)[1][:, :5].reshape(-1)
            # top_2 = torch.topk(probs, k=2)[1]
            # output_tokens[torch.arange(len(sorted_index)), sorted_index] = top_2[:, :, 1][torch.arange(len(sorted_index)), sorted_index]
            # output_tokens[five_by_five+4] = top_2[:, :, 0][five_by_five]
            # _scores = torch.gather(probs, index=output_tokens.unsqueeze(-1), dim=-1).squeeze()
            # output_scores[torch.arange(len(sorted_index)), sorted_index]=_scores[torch.arange(len(sorted_index)), sorted_index]
            

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        ), _tokens


@register_model_architecture("cmlm_transformer", "cmlm_transformer")
def cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)
