import copy
import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaModel, PreTrainedTokenizerFast
from transformers import RobertaModel, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from tokenizers.implementations.sentencepiece_bpe import SentencePieceBPETokenizer

from zsmt.bert_seq2seq import BertDecoderModel, BertEncoderModel, BertOutputLayer, BertConfig
from zsmt.textprocessor import TextProcessor, add_lang_special_toks


def decoder_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int, layer: int = 6,
                   embed_dim: int = 768, intermediate_dim: int = 3072) -> PretrainedConfig:
    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": embed_dim,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_dim,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": layer,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "is_decoder": True,
    }
    config = BertConfig(**config)
    config.add_cross_attention = True
    return config

def roberta_config(vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(
        'roberta-large',
        is_decoder=True,
        add_cross_attention=True,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )
    return config

def future_mask(tgt_mask):
    attn_shape = (tgt_mask.size(0), tgt_mask.size(1), tgt_mask.size(1))
    future_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type_as(tgt_mask)
    return ~future_mask & tgt_mask.unsqueeze(-1)

class Seq2Seq(nn.Module):
    def __init__(self, dst_text_processor: TextProcessor, dec_layer: int = 3, embed_dim: int = 768,
                 intermediate_dim: int = 3072,
                 freeze_encoder: bool = False, freeze_decoder: bool = False,
                 shallow_encoder: bool = False,
                 multi_stream: bool = False, pretrained_decoder: str = '',
                 srct_text_processor: TextProcessor = None, src_lang_fam: bool = False,
                 xlm_name: str = 'xlm-roberta-base'):
        super(Seq2Seq, self).__init__()
        self.text_processor = dst_text_processor
        self.srct_text_processor = srct_text_processor
        if not self.text_processor:
            self.text_processor = dst_text_processor

        dst_vocab_size = dst_text_processor.tokenizer.get_vocab_size()
        self.config_dec = decoder_config(vocab_size=dst_vocab_size,
                                        pad_token_id=dst_text_processor.pad_token_id(),
                                        bos_token_id=dst_text_processor.bos_token_id(),
                                        eos_token_id=dst_text_processor.sep_token_id(),
                                        layer=dec_layer, embed_dim=embed_dim, intermediate_dim=intermediate_dim)
        # TODO: integrate these two functions
        # self.config_dec = roberta_config(vocab_size=dst_vocab_size,
        #                                  pad_token_id=dst_text_processor.pad_token_id(),
        #                                  bos_token_id=dst_text_processor.bos_token_id(),
        #                                  eos_token_id=dst_text_processor.sep_token_id(),)
        if srct_text_processor is not None:
            self.config_enc = decoder_config(vocab_size=srct_text_processor.tokenizer.get_vocab_size(),
                                            pad_token_id=srct_text_processor.pad_token_id(),
                                            bos_token_id=srct_text_processor.bos_token_id(),
                                            eos_token_id=srct_text_processor.sep_token_id(),
                                            layer=dec_layer, embed_dim=embed_dim, intermediate_dim=intermediate_dim)
        else:
            self.config_enc = copy.deepcopy(self.config_dec)
        self.config_enc.add_cross_attention = False
        self.config_enc.is_decoder = False

        self.multi_stream = multi_stream
        self.use_xlm = not shallow_encoder
        self.pretrained_decoder = pretrained_decoder
        self.src_lang_fam = src_lang_fam

        if self.use_xlm:
            tokenizer_class, weights, model_class = XLMRobertaTokenizer, xlm_name, XLMRobertaModel
            self.input_tokenizer = tokenizer_class.from_pretrained(weights)
            self.encoder = model_class.from_pretrained(weights)
            if src_lang_fam:
                add_lang_special_toks(self.input_tokenizer)
                self.encoder.resize_token_embeddings(len(self.input_tokenizer))
        else:
            self.encoder = BertEncoderModel(self.config_enc)

        if self.multi_stream:
            self.shallow_encoder = BertEncoderModel(self.config_enc)
            self.encoder_gate = nn.Parameter(torch.zeros(1, self.config_enc.hidden_size).fill_(0.1),
                                             requires_grad=True)

        self.dec_layer = dec_layer
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim

        if self.pretrained_decoder == '':
            self.decoder = BertDecoderModel(self.config_dec)
        elif self.pretrained_decoder:
            model_name_dec = self.pretrained_decoder
            self.config_dec = AutoConfig.from_pretrained(model_name_dec)
            self.config_dec.is_decoder = True
            self.config_dec.add_cross_attention = True
            self.decoder = RobertaModel.from_pretrained(model_name_dec, config=self.config_dec)

        dst_tok = dst_text_processor.tokenizer
        if isinstance(dst_tok, PreTrainedTokenizerFast) and 'xlm-roberta' in dst_tok.name_or_path:
            print('for decoder embeddings, discarding pretrained roBERTa embeddings, tying to XLM-R instead')
            num_embed = self.encoder.embeddings.word_embeddings.num_embeddings
            self.config_dec.vocab_size = num_embed
            self.decoder.embeddings = RobertaEmbeddings(self.config_dec)
            self.decoder._tie_encoder_decoder_weights(
                self.decoder.embeddings, self.encoder.embeddings, self.decoder.base_model_prefix)
            # since tied, if either is frozen, then both are frozen
            freeze_decoder = freeze_encoder = freeze_encoder or freeze_decoder
        elif isinstance(dst_tok, SentencePieceBPETokenizer):
            pt_vocab_size = self.config_dec.vocab_size
            print(f'using new vocab (n={dst_vocab_size}) insead of from roBERTa (n={pt_vocab_size})')
            num_embed = dst_vocab_size
            self.config_dec.vocab_size = num_embed
            self.decoder.embeddings = RobertaEmbeddings(self.config_dec)

        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = not freeze_encoder
        for param in self.decoder.embeddings.parameters():
            param.requires_grad = not freeze_decoder

        self.output_layer = BertOutputLayer(self.config_dec)
        if self.multi_stream:
            self.shallow_encoder._tie_or_clone_weights(self.output_layer,
                                                       self.shallow_encoder.embeddings.position_embeddings)

            self.shallow_encoder._tie_or_clone_weights(self.shallow_encoder.embeddings.position_embeddings,
                                                       self.decoder.embeddings.position_embeddings)
            self.shallow_encoder._tie_or_clone_weights(self.shallow_encoder.embeddings.token_type_embeddings,
                                                       self.decoder.embeddings.token_type_embeddings)
            if not self.pretrained_decoder:
                self.shallow_encoder._tie_or_clone_weights(self.shallow_encoder.embeddings.word_embeddings,
                                                           self.decoder.embeddings.word_embeddings)

    def src_eos_id(self):
        if self.use_xlm:
            return self.input_tokenizer.eos_token_id
        else:
            return self.text_processor.sep_token_id()

    def src_pad_id(self):
        if self.use_xlm:
            return self.input_tokenizer.pad_token_id
        else:
            return self.text_processor.pad_token_id()

    def decode_src(self, src, remove_first=False, cleanup=True):
        if remove_first:
            src = src[1:]
        if self.use_xlm:
            return self.input_tokenizer.decode(src, clean_up_tokenization_spaces=cleanup)
        else:
            return self.text_processor.tokenizer.decode(src.numpy(), clean_up_tokenization_spaces=cleanup)

    def encode(self, src_inputs, src_mask, srct_inputs, srct_mask):
        """
        srct_inputs is used in case of multi_stream where srct_inputs is the second stream.
        """
        device = self.encoder.device
        if src_inputs.device != device:
            src_inputs = src_inputs.to(device)
            src_mask = src_mask.to(device)
        encoder_states = self.encoder(src_inputs, attention_mask=src_mask)

        if self.use_xlm:
            encoder_states = encoder_states['last_hidden_state']

        if self.multi_stream:
            if srct_inputs.device != device:
                srct_inputs = srct_inputs.to(device)
                srct_mask = srct_mask.to(device)
            shallow_encoder_states = self.shallow_encoder(srct_inputs, attention_mask=srct_mask)
            return encoder_states, shallow_encoder_states
        else:
            return encoder_states, None

    def forward(self, src_inputs, tgt_inputs, src_mask, tgt_mask, srct_inputs, srct_mask, log_softmax: bool = False):
        """
        srct_inputs is used in case of multi_stream where srct_inputs is the second stream.
        """
        "Take in and process masked src and target sequences."
        device = self.encoder.embeddings.word_embeddings.weight.device
        src_inputs = src_inputs.to(device)
        if tgt_inputs.device != device:
            tgt_inputs = tgt_inputs.to(device)
            tgt_mask = tgt_mask.to(device)
        if src_mask.device != device:
            src_mask = src_mask.to(device)

        if self.multi_stream and srct_inputs.device != device:
            srct_inputs = srct_inputs.to(device)
        if self.multi_stream and srct_mask.device != device:
            srct_mask = srct_mask.to(device)
        encoder_states, shallow_encoder_states = self.encode(src_inputs, src_mask, srct_inputs=srct_inputs,
                                                             srct_mask=srct_mask)

        subseq_mask = future_mask(tgt_mask[:, :-1])
        if subseq_mask.device != tgt_inputs.device:
            subseq_mask = subseq_mask.to(device)
        decoder_output = self.attend_output(encoder_states, shallow_encoder_states, src_mask, srct_mask, subseq_mask,
                                            tgt_inputs[:, :-1])
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
        tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def attend_output(self, encoder_states, shallow_encoder_states, src_mask, srct_mask, tgt_attn_mask, tgt_inputs):
        if self.multi_stream:
            if isinstance(self.decoder, BertDecoderModel):
                first_decoder_output = self.decoder(encoder_states=encoder_states, input_ids=tgt_inputs,
                                                    encoder_attention_mask=src_mask, tgt_attention_mask=tgt_attn_mask)
                second_decoder_output = self.decoder(encoder_states=shallow_encoder_states, input_ids=tgt_inputs,
                                                 encoder_attention_mask=srct_mask, tgt_attention_mask=tgt_attn_mask)
            else:
                first_decoder_output = self.decoder(
                    input_ids=tgt_inputs,
                    attention_mask=tgt_attn_mask,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=src_mask
                )[0]
                second_decoder_output = self.decoder(
                    input_ids=tgt_inputs,
                    attention_mask=tgt_attn_mask,
                    encoder_hidden_states=shallow_encoder_states,
                    encoder_attention_mask=srct_mask
                )[0]
            eps = 1e-7
            sig_gate = torch.sigmoid(self.encoder_gate + eps)
            decoder_output = sig_gate * first_decoder_output + (1 - sig_gate) * second_decoder_output
        else:
            if isinstance(self.decoder, BertDecoderModel):
                decoder_output = self.decoder(encoder_states=encoder_states, input_ids=tgt_inputs,
                                          encoder_attention_mask=src_mask, tgt_attention_mask=tgt_attn_mask)
            else:
                decoder_output = self.decoder(
                    input_ids=tgt_inputs,
                    attention_mask=tgt_attn_mask,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=src_mask
                )[0]
        return decoder_output

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump((self.dec_layer, self.embed_dim, self.intermediate_dim, self.freeze_encoder,
                self.multi_stream), fp)
        try:
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        except:
            torch.cuda.empty_cache()
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        finally:
            torch.cuda.empty_cache()

    @staticmethod
    def load(cls, out_dir: str, tok_dir: str = '', pretrained_decoder: str = '', src_lang_fam=False, load_output=True,
             xlm_name: str = 'xlm-roberta-base', xlmr_dst_tok=False,
             freeze_encoder=False, freeze_decoder=False):
        srct_text_processor, dst_text_processor = None, None

        if tok_dir:
            srct_text_processor = TextProcessor(tok_model_path=tok_dir)

        if xlmr_dst_tok:
            dst_text_processor = TextProcessor(pretrained_name='xlm-roberta-base')
        elif pretrained_decoder and tok_dir:
            dst_text_processor = src_lang_fam
        elif pretrained_decoder:
            dst_text_processor = TextProcessor(pretrained_name=pretrained_decoder)
        if dst_text_processor is None and srct_text_processor is None:
            print(f'ERROR: must pass in at least one of --pt-dec or --tok')
            sys.exit(-1)
        dst_text_processor = dst_text_processor or srct_text_processor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            unpickled_obj = pickle.load(fp)
            dec_layer, embed_dim, intermediate_dim, _, multi_stream = unpickled_obj

        srct_text_processor = srct_text_processor if multi_stream else None

        mt_model = cls(dst_text_processor=dst_text_processor, dec_layer=dec_layer, embed_dim=embed_dim,
                        intermediate_dim=intermediate_dim, multi_stream=multi_stream,pretrained_decoder=pretrained_decoder,
                        srct_text_processor=srct_text_processor, src_lang_fam=src_lang_fam,
                        freeze_encoder=freeze_encoder, freeze_decoder=freeze_decoder, xlm_name=xlm_name)
        state_dict = torch.load(os.path.join(out_dir, "mt_model.state_dict"), map_location=device)
        if not load_output:
            print('not loading weights for `decoder.embeddings` and `output_layer`...')
            saved_keys = {}
            for k, v in state_dict.items():
                if k.startswith('decoder.embeddings') or k.startswith('output_layer'):
                    if 'LayerNorm' in k:
                        continue
                    saved_keys[k] = v
            for k in saved_keys:
                del state_dict[k]

        mt_model.load_state_dict(state_dict,
                                    strict=False)
        if not load_output: # TODO: is this necessary?
            mt_model.decoder = mt_model.decoder.to(device)
            mt_model.encoder = mt_model.encoder.to(device)

        return mt_model
