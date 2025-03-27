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
from fairseq.models.nat import CMLMNATransformerModel
from fairseq.models.nat.cmlm_transformer import _skeptical_unmasking, _random_mask
from fairseq.utils import new_arange
import torch.nn.functional as F
import torch


def encoder_random_mask(src_tokens, src_dict):
    pad = src_dict.pad()
    bos = src_dict.bos()
    eos = src_dict.eos()
    unk = src_dict.unk()

    src_masks = (
        src_tokens.ne(pad) & src_tokens.ne(bos) & src_tokens.ne(eos) 
    )
   
    src_score = src_tokens.clone().float().uniform_()
    src_score.masked_fill_(~src_masks, 2.0)
    src_length = src_masks.sum(1).float()
    src_length =  1+0.1 * src_length * src_length.clone().uniform_() # At most 10 percent mask, directly from jmnat not adding arg 
                                                                     # at least one masked cuz multi GPU sux

    _, src_rank = src_score.sort(1)
    src_cutoff = new_arange(src_rank) < src_length[:, None].long()
    src_masked = src_tokens.masked_fill(
        src_cutoff.scatter(1, src_rank, src_cutoff), unk
    )
    return src_masked



@register_model("cmlmc_transformer")
class CMLMCNATransformerModel(CMLMNATransformerModel):
    @staticmethod
    def add_args(parser):
        CMLMNATransformerModel.add_args(parser)
        ######################################## CMLMC arguments ####################################################
        parser.add_argument('--selfcorrection', type=int, default=-1,
                            help='starting from selfcorrection step, use model to generate tokens for the currently'
                                 'unmasked positions (likely wrong), and teach the model to correct it to the right token,'
                                 'aimed at doing beam-search like correction')
        parser.add_argument("--replacefactor", type=float, default=0.30,
                            help="percentage of ground truth tokens replaced during SelfCorrection or GenerationSampling")
        parser.add_argument('--onesteploss', default=False, action='store_true',
                            help='compute and optimize for the loss of one step prediction, from fully masked input')
        ######################################## CMLMC arguments ####################################################
        parser.add_argument('--encoder-mask', default=False, action='store_true',
                            help='JmNAT style encoder masking')
        parser.add_argument('--encoder-pred', default=False, action='store_true',
                            help='JmNAT style encoder token prediction')


    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        ######################################## CMLMC Modifications ####################################################
        self.selfcorrection = args.selfcorrection
        self.correctingself = False
        self.replacefactor = args.replacefactor
        self.onesteploss = args.onesteploss
        ######################################## CMLMC Modifications ####################################################
        self.encoder_mask = args.encoder_mask
        self.encoder_pred = args.encoder_pred

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, update_num=None, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        src_masked = encoder_random_mask(src_tokens, self.encoder.dictionary)
        encoder_out = self.encoder(src_masked if self.encoder_mask else src_tokens, src_lengths=src_lengths, **kwargs)

        if self.selfcorrection != -1:
            # if update_num is not None:
            self.correctingself = True

        if getattr(self.args, 'ctc', False):
            best_alignment, masked_input_out, force_emits = self.find_best_align(src_tokens, tgt_tokens, encoder_out)
            prev_output_tokens=best_alignment.clone()
            force_emits_cor=force_emits.clone()
            prev_output_tokens, force_emits = _random_mask(prev_output_tokens, self.decoder.dictionary, force_emits)

        word_ins_out, inner_states = self.decoder(normalize=False, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, inner_states=True)
        embed_states=inner_states['inner_states']
        word_ins_mask = prev_output_tokens.eq(self.unk)

        valid_token_mask = (prev_output_tokens.ne(self.pad) &
                            prev_output_tokens.ne(self.bos) &
                            prev_output_tokens.ne(self.eos))
        revealed_token_mask = (prev_output_tokens.ne(self.pad) &
                                prev_output_tokens.ne(self.bos) &
                                prev_output_tokens.ne(self.eos) &
                                prev_output_tokens.ne(self.unk))
        ######################################## CMLMC Modifications ####################################################
        if not getattr(self.args, 'ctc', False):
            masked_input_out = self.decoder(normalize=False,
                                            prev_output_tokens=tgt_tokens.masked_fill(valid_token_mask, self.unk),
                                            encoder_out=encoder_out)

        revealed_length = revealed_token_mask.sum(-1).float()
        replace_length = revealed_length * self.replacefactor

        # ############## partial self correction, least confident replacing #############################
        # sample the fully_masked_input's output, pick the one with the highest probability
        masked_input_out_scores, masked_input_out_tokens = F.log_softmax(masked_input_out, -1).max(-1)
        # ############################################################################
        # ############## the following line implements random replacing by re-sampling the scores from U(0, 1)
        # ############## if the next line is commented out, least confident replacing is used
        masked_input_out_scores.uniform_()
        # ############################################################################

        # Fill any non-revealed position with 2 (0 also works, but use 2.0 so it's compatible with random
        # sampling as well), which is higher than any valid loglikelihood
        masked_input_out_scores.masked_fill_(~revealed_token_mask, 2.0)
        # calculate the number of tokens to be replaced for each sentence, from which to learn self-correction

        # sort the fully_masked_input's output based on confidence,
        # generate the replaced token mask on the least confident 15%
        _, replace_rank = masked_input_out_scores.sort(-1)
        replace_token_cutoff = new_arange(replace_rank) < replace_length[:, None].long()
        replace_token_mask = replace_token_cutoff.scatter(1, replace_rank, replace_token_cutoff)

        # replace the corresponding tokens in the noisy input sentence, with the token from the generated output
        replaced_input_tokens = prev_output_tokens.clone()
        replaced_input_tokens[replace_token_mask] = masked_input_out_tokens[replace_token_mask]
        if getattr(self.args, 'ctc', False):
            force_emits_cor[replace_token_mask]=-1

        replace_input_out, replace_inner_states = self.decoder(normalize=False,
                                            prev_output_tokens=replaced_input_tokens,
                                            encoder_out=encoder_out,
                                            inner_states=True)
        replace_embed_states=replace_inner_states['inner_states']

        # ############## Adding output for L_corr calculation #############################
        word_ins_out=torch.cat((word_ins_out, replace_input_out), 0)
        word_ins_mask=torch.cat((word_ins_mask, replace_token_mask), 0)
        embed_states = [torch.cat((embed_states[i], replace_embed_states[i]), 1) for i in range(len(embed_states))]
        tgt_tokens=torch.cat((tgt_tokens, tgt_tokens), dim=0)
        if getattr(self.args, 'ctc', False):
            force_emits=torch.cat((force_emits, force_emits), dim=0)
        # ############## Adding output for L_mask calculation #############################
        return_dict={
            "word_ins":{
                "out": word_ins_out, 
                "tgt": tgt_tokens,
                "mask": word_ins_mask if not getattr(self.args, 'ctc', False) else force_emits, 
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }
        }
        if self.args.layer_prediction_loss_factor>0:
            self.per_layer_loss(embed_states, return_dict, tgt_tokens, word_ins_mask, force_emits)

        if self.encoder_pred:
            self.add_encoder_pred_loss(return_dict, encoder_out, src_tokens, src_masked)
        if getattr(self.args, 'ctc', False):
            return return_dict, torch.cat((best_alignment, best_alignment), dim=0) 
        else:
            self.add_length_loss(return_dict, encoder_out, tgt_tokens[:len(tgt_tokens)//2])
            return return_dict
        ######################################## CMLMC Modifications ####################################################

    def add_encoder_pred_loss(self, return_dict, encoder_out, src_tokens, src_masked):
        encoder_logits = self.decoder.output_layer(encoder_out['encoder_out'][0].transpose(0, 1))
        return_dict['encoder_loss']={
                "out": encoder_logits,
                "tgt": src_tokens ,
                "mask": src_masked.eq(self.unk),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": .01 , #directly from jmnat! not adding arg
            }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # ############### CMLM's original code ##################################################
        # ######## CMLM made the decision of NOT updating revealed token confidences, which is understandable in a way
        # ######## because the model is not trained for these positions, but on the other hand it means it never
        # ######## updates the revealed tokens on purpose...
        # output_masks = output_tokens.eq(self.unk)

        # ################################## CMLMC ##################################################
        # ######## update all valid token positions (other than BOS, EOS, PAD) ####################
        output_masks = output_tokens.ne(self.pad) & output_tokens.ne(self.bos) & output_tokens.ne(self.eos)
        # ################################## CMLMC ##################################################

        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )
            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        ), _tokens


@register_model_architecture("cmlmc_transformer", "cmlmc_transformer")
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


@register_model_architecture("cmlmc_transformer", "cmlmc_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)

@register_model_architecture("cmlmc_transformer", "cmlmc_transformer_iwslt_en_de")
def cmlm_iwslt_en_de(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    cmlm_base_architecture(args)

@register_model_architecture("cmlmc_transformer", "cmlmc_transformer_iwslt_so_tg")
def cmlm_iwslt_so_tg(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    cmlm_base_architecture(args)
