from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import CMLMNATransformerModel
from fairseq.models.nat.cmlm_transformer import _random_mask, _block_mask, _block_unmasking
from collections import namedtuple
import torch
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models.transformer import Embedding
from fairseq.utils import new_arange
from fairseq.modules.ema_module import EMAModule, EMAModuleConfig
from torch.nn.utils.rnn import pad_sequence
from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
from fairseq import utils
from argparse import Namespace
from fairseq.scoring import bleu
from fairseq.criterions.oaxe import OrderAgnostiCrossEntropy

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = torch.round(
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    )
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cmlm_distill")
class CMLMDdistill(CMLMNATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        try:
            if not args.random_init_decoder:
                self.decoder.load_state_dict({key.replace('decoder.', ''): val for key, val in torch.load(args.teacher_path)['model'].items() if key.startswith('decoder')})
                if not getattr(args, 'ctc', False):
                    self.decoder.embed_length = Embedding(256, self.decoder.encoder_embed_dim, None)

            if not args.random_init_encoder:
                self.encoder.load_state_dict({key.replace('encoder.', ''): val for key, val in torch.load(args.teacher_path)['model'].items() if key.startswith('encoder')})
                if not getattr(args, 'ctc', False):
                    self.decoder.embed_length.load_state_dict({key.replace('decoder.embed_length.', ''): val for key, val in torch.load(args.teacher_path)['model'].items() if key.startswith('decoder.embed_length')})
            else:
                self.args.optimize_encoder = True
            
            DummyTask = namedtuple('DummyTask', 'source_dictionary target_dictionary')
            dummy_task = DummyTask(source_dictionary=encoder.dictionary, target_dictionary=decoder.dictionary)

            self.teacher = [CMLMNATransformerModel.build_model(args, dummy_task).cuda().half()]
            self.teacher[0].load_state_dict(torch.load(args.teacher_path)['model'])
            if args.teacher_ema:
                teacher_ema_config=EMAModuleConfig()
                teacher_ema_config.ema_decay=args.teacher_ema_decay
                teacher_ema_config.ema_fp32=True
                self.teacher_ema=EMAModule(self.teacher[0], teacher_ema_config, device='cuda')
        except:
            pass
        if args.revealed_loss:
            assert args.mid_mask_policy=='discrete'
        if args.mid_mask_policy=='discrete':
            assert args.step_count!=-1
        if args.step_weight_update>0:
            assert args.mid_mask_policy=='discrete'

        self.geneator = IterativeRefinementGenerator(
            self.decoder.dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 0),
            beam_size=getattr(args, "iter_decode_with_beam", 5),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", True),
            retain_history=getattr(args, "retain_iter_history", False),
            bleu_reranker=getattr(args, "bleu_reranker", False), 
            deduplicate=True
        )

        self.policy_to_max = {
            'half': lambda sentence_len: sentence_len,
            'uniform': lambda sentence_len: sentence_len,
            'discrete': lambda sentence_len: torch.ones_like(sentence_len, device=sentence_len.device)*self.args.step_count*2
        }
        self.policy_to_mid = {
            'half': lambda sentence_len, unk_count, stp_cnt: [(sentence_len - i*unk_count/stp_cnt).long() for i in range(stp_cnt-1, -1, -1)],
            'uniform': lambda sentence_len, unk_count:1 + (torch.rand(len(sentence_len), device=sentence_len.device)*unk_count + sentence_len - unk_count).long(),
            'discrete': lambda sentence_len, unk_count: 1 + torch.floor(((sentence_len - unk_count)*self.args.step_count*2)/sentence_len).long(),
        }

        self.block_policy_to_mid = {
            'half': lambda blk_size, unk_count_per_block, stp_cnt: [(blk_size-i*unk_count_per_block/stp_cnt).long() for i in range(stp_cnt-1, -1, -1)],
            'full': lambda blk_size, unk_count_per_block, stp_cnt: [blk_size-i for i in range(unk_count_per_block-1, -1, -1)],
            'discrete': lambda blk_size, unk_count_per_block, stp_cnt: [blk_size-(unk_count_per_block-i) for i in range(1, stp_cnt+1)],
            'discrete_all': lambda blk_size, unk_count_per_block, stp_cnt: [blk_size-(unk_count_per_block-1), blk_size]
        }

        for name, param in self.encoder.named_parameters():
            param.requires_grad=args.optimize_encoder
        
        if not getattr(args, 'ctc', False):
            self.decoder.embed_length.weight.requires_grad=args.optimize_length_predictor
        if getattr(args, 'insertCausalSelfAttn', False):
            assert getattr(args, 'no_scale_embedding', False)


        if self.args.oaxe_distill_loss_factor>0 or self.args.oaxe_orig_loss_factor>0.:
            args.label_smoothing=0.1
            args.skip=0.15
            self.oaxe = OrderAgnostiCrossEntropy(args, dummy_task)
            assert args.embed_loss_factor==0


    @staticmethod
    def add_args(parser):
        super(CMLMDdistill, CMLMDdistill).add_args(parser)
        parser.add_argument('--teacher-path', type=str, metavar='STR',
            help='path to the trained teacher for distillation.')
        parser.add_argument('--optimize-encoder', default=False, action='store_true',
            help='optimize the encoder of student.')
        parser.add_argument('--random-init-encoder', default=False, action='store_true',
            help='Initialize encoder randomly.')
        parser.add_argument('--random-init-decoder', default=False, action='store_true',
            help='Initialize decoder randomly.')
        parser.add_argument('--hard-label', default=False, action='store_true',
            help='use hard labels for distillation.')
        parser.add_argument('--step-count', type=int, default=-1,
            help='Distill for a specific number of steps.')
        parser.add_argument('--mid-mask-policy', type=str, default='half',
            help='Middle masking policy: half, uniform, discrete')
        parser.add_argument('--revealed-loss', default=False, action='store_true',
            help='Compute loss only for revealed tokens.')
        parser.add_argument('--step-weight-update', type=float, default=0.,
            help='Update weight for steps ema.')
        parser.add_argument('--step-weight-temp', type=float, default=1.,
            help='Temperature for step sampling.')
        parser.add_argument('--teacher-ema', default=False, action='store_true',
            help='Run ema for teacher.')
        parser.add_argument('--teacher-ema-decay', default=.9997, type=float,
            help='Teacher ema decay.')
        parser.add_argument('--optimize-length-predictor', default=False, action='store_true',
            help='Optimize length predictor.')
        parser.add_argument('--beam-length', default=False, action='store_true',
            help='Use beam search predicted length to optimize length predictor.')
        parser.add_argument('--beam-sample', default=False, action='store_true',
            help='Use beam search predicted samples to optimize decoder.')
        parser.add_argument('--embed-loss-factor', default=0., type=float,
            help='Embed Loss Factor.')
        parser.add_argument('--embed-loss-layers', default=1, type=int,
            help='Number of layers with embed loss.')
        parser.add_argument('--cross-attn-loss-factor', default=0., type=float,
            help='Attention between decoder and encoder Loss Factor.')
        parser.add_argument('--self-attn-loss-factor', default=0., type=float,
            help='Decoder attention Loss Factor.')
        parser.add_argument('--orig-loss-factor', default=0., type=float,
            help='Original Loss Factor.')
        parser.add_argument('--oaxe-orig-loss-factor', default=0., type=float,
            help='OAXE Original Loss Factor.')
        parser.add_argument('--oaxe-distill-loss-factor', default=0., type=float,
            help='OAXE Distill Loss Factor.')
        parser.add_argument('--distill-loss-factor', default=1., type=float,
            help='Distill Loss Factor.')
        parser.add_argument('--std-temp', default=1., type=float,
            help='student temperature for distillation.')
        parser.add_argument('--teacher-temp', default=1., type=float,
            help='Minimum teacher temperature for distillation.')
        parser.add_argument('--correction-loss-factor', default=0., type=float,
            help='Correction Loss Factor.')
        parser.add_argument('--bleu-reranker', default=False, action='store_true',
            help='rank beam sentences based on bleu.')
        parser.add_argument('--teacher-beam-size', default=1, type=int,
            help='Beam size for teachers generation.')
        parser.add_argument('--distill-on-valid', default=False, action='store_true',
            help='Use validation set for distillation.')
        parser.add_argument('--teacher-iterative-steps', default=2, type=int,
            help='Number of teacher steps.')
        parser.add_argument('--teacher-random-unmask', default=False, action='store_true',
            help='For middle step instead of skeptical unmasking, randomly mask.')
        parser.add_argument('--unsupervised', default=False, action='store_true',
            help='Do not use targets for distillation.')
        parser.add_argument('--ctc-distill', default=False, action='store_true',
            help='Distill ctc model.')
        
    def forward_decoder_teacher(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step
        assert not isinstance(step, int)
        assert not isinstance(max_step, int)

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks=kwargs['mask']
        #output_masks = output_tokens.eq(self.unk)
        #if getattr(self.args, 'insertCausalSelfAttn', False):
        #    output_masks = output_tokens.ne(self.pad) & output_tokens.ne(self.bos) & output_tokens.ne(self.eos)

        probs, teacher_inner_states = self.teacher[0].decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            inner_states=True,
        )
        #if all(step==max_step):
        probs = F.log_softmax(probs, -1)
        _tokens = probs.argmax(-1)
        #else:
        #    probs = F.log_softmax(probs/1.2, -1)
        #    _tokens = torch.distributions.categorical.Categorical(probs=torch.exp(probs), validate_args=False).sample()
        _scores = torch.gather(probs, index=_tokens.unsqueeze(-1), dim=-1).squeeze(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())


        if getattr(self.args, 'mask_policy', 'uniform')=='uniform':
            skeptical_mask = _skeptical_unmasking(
                output_scores if not self.args.teacher_random_unmask else -(1+torch.rand(*output_scores.shape, device=output_scores.device))*(output_masks), 
                output_tokens.ne(self.pad), 
                (1 - step / max_step).unsqueeze(-1)
            )
        else:
            skeptical_mask = _block_unmasking(
                output_scores if not self.args.teacher_random_unmask else -(1+torch.rand(*output_scores.shape, device=output_scores.device))*(output_masks), 
                output_tokens.ne(self.pad), 
                (1 - step / max_step).unsqueeze(-1),
                max_step
            )
        skeptical_mask[step>=max_step, :] = False
        output_tokens.masked_fill_(skeptical_mask, self.unk)
        output_scores.masked_fill_(skeptical_mask, 0.0)




        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        ), probs, teacher_inner_states['inner_states'], teacher_inner_states['cross-attn'], teacher_inner_states['self-attn'], _tokens


    @torch.no_grad()
    def pick_best_bleu_index(self, finalized, tgt):
        scorer = bleu.Scorer(self)
        refs = tgt.unsqueeze(1).expand(-1, self.args.teacher_beam_size, -1).reshape(-1, tgt.shape[-1])
        bleu_scores = []
        for hyp,ref in zip(finalized, refs):
            ref = torch.IntTensor(ref.int().cpu())
            scorer.add(ref, torch.IntTensor(hyp.int().cpu()))
            bleu_scores.append(torch.tensor(scorer.score()))
            scorer.reset()
        bleu_scores = torch.tensor(bleu_scores, device=tgt.device)
        offset = bleu_scores.reshape(-1, self.args.teacher_beam_size).argmax(-1)
        return torch.arange(len(tgt), device=tgt.device)*self.args.teacher_beam_size+offset

    def index_tensors(self, encoder_out, decoder_out, indices):
        encoder_out = self.encoder.reorder_encoder_out(
            encoder_out, indices
        )
        
        decoder_out = decoder_out._replace(
            output_tokens = decoder_out.output_tokens[indices],
            output_scores = decoder_out.output_scores[indices],
        )
        return encoder_out, decoder_out 

    def compute_bleu(self, gen_out, targets):
        import sacrebleu
        def decode(toks, escape_unk=False):
            s = self.decoder.dictionary.string(
                toks.int().cpu()
            )
            return s

        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(utils.strip_pad(gen_out[i], self.decoder.dictionary.pad())))
            refs.append(
                decode(
                    utils.strip_pad(targets[i], self.decoder.dictionary.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        #print(hyps[0])
        #print(refs[0])
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")

    def generate_oaxe_net_output(self, word_ins_out, word_ins_mask, encoder_out, tgt_tokens):
        length_out = self.decoder.forward_length(
                normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )    
        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }
    
    def generate_synthetic(self, sample, encoder_out):
        self.geneator.max_iter = 8
        hypos = self.geneator.generate(self.teacher, sample, encoder_out=encoder_out)
        hypos = list(map(lambda x: x[0]['tokens'], hypos))
        prev_output_tokens=pad_sequence(hypos, padding_value=self.pad, batch_first=True)
        if getattr(self.args, 'ctc', False):
            return prev_output_tokens
        revealed_token_mask = (prev_output_tokens.ne(self.pad) &
                                prev_output_tokens.ne(self.bos) &
                                prev_output_tokens.ne(self.eos))
        
        ratio = torch.rand((len(prev_output_tokens),), device=prev_output_tokens.device)
        replace_length = revealed_token_mask.sum(-1).float() * ratio

        masked_input_out_scores = torch.zeros_like(prev_output_tokens, device=prev_output_tokens.device).float()
        masked_input_out_scores.uniform_()
        masked_input_out_scores.masked_fill_(~revealed_token_mask, 2.0)
        _, replace_rank = masked_input_out_scores.sort(-1)
        replace_token_cutoff = new_arange(replace_rank) < replace_length[:, None].long()
        replace_token_mask = replace_token_cutoff.scatter(1, replace_rank, replace_token_cutoff)
        prev_output_tokens[replace_token_mask]=self.unk
        return prev_output_tokens
    
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        if self.args.teacher_ema and self.training:
            self.teacher_ema.step(self)
            self.teacher[0]=self.teacher_ema.reverse(self.teacher[0])
        self.teacher[0].eval()
        self.eval()
        
        with torch.no_grad():
            encoder_out = self.teacher[0].encoder(src_tokens, src_lengths=src_lengths, **kwargs)
            
            unsup_tgt=None
            force_emits=None
            if self.args.unsupervised:
                sample={'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}, "target": tgt_tokens}
                if getattr(self.args, 'ctc', False):
                    prev_output_tokens = self.generate_synthetic(sample, encoder_out)
                    best_alignemnt = prev_output_tokens.clone()
                    blank_mask = best_alignemnt.eq(self.blank)
                    blank_emits = torch.cumsum(blank_mask, dim=-1)*2-2
                    no_blank_emits = torch.cumsum(~blank_mask, dim=-1)*2-1
                    force_emits = blank_emits*blank_mask + no_blank_emits*(~blank_mask)
                    force_emits[best_alignemnt.eq(self.pad)] = self.pad
                    print('unsupervised')
                    #unsup_tgt=[tgt.unique_consecutive() for tgt in unsup_tgt] 
                    #unsup_tgt=pad_sequence([tgt[tgt!=self.blank] for tgt in unsup_tgt], padding_value=self.pad, batch_first=True)
                else:
                    prev_output_token = self.generate_synthetic(sample, encoder_out)

            if getattr(self.args, 'ctc_distill', False):
                self.teacher[0].blank = self.blank
                if not self.args.unsupervised:
                    best_alignemnt, _, force_emits = self.teacher[0].find_best_align(src_tokens, tgt_tokens if unsup_tgt is None else unsup_tgt, encoder_out)
                    prev_output_tokens = best_alignemnt.clone()

                force_emits[best_alignemnt.ne(self.pad)] = -1
                if getattr(self.args, 'mask_policy', 'uniform')=='uniform':
                    prev_output_tokens, force_emits =_random_mask(prev_output_tokens, self.decoder.dictionary, force_emits=force_emits, full_mask=getattr(self.args, 'noise', 'random_mask')=='full_mask')
                else:
                    blk_size = 2**torch.randint(low=self.args.teacher_iterative_steps, high=5, size=(1,), device=prev_output_tokens.device)
                    if self.args.mid_mask_policy=='full':
                        unk_count_per_block = torch.randint(low=self.args.teacher_iterative_steps, high=blk_size.item()+1, size=(1,), device=prev_output_tokens.device)
                    else:
                        unk_count_per_block = torch.randint(low=self.args.teacher_iterative_steps, high=blk_size.item()+1, size=(len(prev_output_tokens),), device=prev_output_tokens.device)
                    if getattr(self.args, 'noise', 'random_mask')=='full_mask':
                        unk_count_per_block = torch.ones_like(unk_count_per_block)*blk_size

                    prev_output_tokens, force_emits =_block_mask(prev_output_tokens, self.decoder.dictionary, force_emits=force_emits, blk_size=blk_size, unk_cnt=unk_count_per_block)
                    



            word_ins_mask = prev_output_tokens.eq(self.unk)
            if getattr(self.args, 'ctc_distill', False):
                word_ins_mask = prev_output_tokens.ne(self.pad)


            if self.args.beam_length or self.args.beam_sample:
                sample={'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}, "target": tgt_tokens}
                hypos=self.geneator.generate(self.teacher, sample, encoder_out=encoder_out)
                hypos=list(map(lambda x: x[0]['tokens'], hypos))
                beam_tokens=pad_sequence(hypos, padding_value=self.pad, batch_first=True)
                beam_length_tgt=self.decoder.forward_length_prediction(None, encoder_out, beam_tokens)
                if self.args.beam_sample:
                    tgt_tokens=beam_tokens
                    prev_output_tokens=_random_mask(beam_tokens, self.decoder.dictionary)
                del sample

            prev_decoder_out = DecoderOut(
                output_tokens=torch.clone(prev_output_tokens),
                output_scores=torch.zeros_like(prev_output_tokens, dtype=torch.float16),
                attn=None,
                step=0,
                max_step=None,
                history=None,
            )
        
            if self.args.teacher_beam_size>1:
                beam_order = (
                    utils.new_arange(prev_output_tokens, self.args.teacher_beam_size, len(prev_output_tokens)).t().reshape(-1)
                )

                encoder_out, prev_decoder_out=self.index_tensors(encoder_out, prev_decoder_out, beam_order)
                assert prev_decoder_out.output_tokens.shape[0]==len(prev_output_tokens)*self.args.teacher_beam_size
                assert encoder_out['encoder_out'][0].shape[1]==len(prev_output_tokens)*self.args.teacher_beam_size

            unk_count = torch.sum(prev_decoder_out.output_tokens.eq(self.unk), dim=-1)
            sentence_len = (torch.sum(prev_decoder_out.output_tokens.ne(self.pad), dim=-1))
            if not getattr(self.args, 'ctc', False):
                sentence_len -= 2
            beam_num = utils.new_arange(src_tokens, len(prev_output_tokens), self.args.teacher_beam_size).reshape(-1)
            if getattr(self.args, 'mask_policy', 'uniform')=='uniform':
                step_schedule = self.policy_to_mid[self.args.mid_mask_policy](sentence_len, unk_count, self.args.teacher_iterative_steps)
            else:
                step_schedule = self.block_policy_to_mid[self.args.mid_mask_policy](blk_size*torch.ones(len(prev_output_tokens), device=prev_output_tokens.device).long(), unk_count_per_block, self.args.teacher_iterative_steps)

            max_step = step_schedule[-1]
            prev_decoder_out = prev_decoder_out._replace(max_step=max_step)

            reveal_step = max_step
            if self.args.revealed_loss:
                reveal_step= mid_step + 1

            for step_num, step in enumerate(step_schedule):
                prev_decoder_out = prev_decoder_out._replace(step=step)
                update_mask=(prev_output_tokens.eq(self.unk) if not getattr(self.args, 'insertCausalSelfAttn', False) else word_ins_mask)
                prev_decoder_out, probs, teacher_embds_new, teacher_cross_attn, teacher_self_attn, teacher_tokens_temp = self.forward_decoder_teacher(
                    prev_decoder_out, encoder_out, mask=update_mask
                )
                if step_num==0:
                    prev_teacher_tokens=teacher_tokens_temp
                    corrupted_tokens=prev_decoder_out.output_tokens.clone()
                    teacher_embds=teacher_embds_new
                for k in range(len(teacher_embds)):
                    teacher_embds[k][update_mask.t()] = teacher_embds_new[k][update_mask.t()]

                next_unk_count = torch.sum(prev_decoder_out.output_tokens.eq(self.unk), dim=-1)
                assert all(unk_count >= next_unk_count)
                unk_count = next_unk_count
                
                teacher_tgt = None
                if getattr(self.args, 'ctc', False):
                    teacher_tokens=prev_output_tokens.clone()
                    teacher_tokens[prev_output_tokens.eq(self.unk)]=teacher_tokens_temp[prev_output_tokens.eq(self.unk)]
                    teacher_tgt=[tgt.unique_consecutive() for tgt in (teacher_tokens if step_num==0 else prev_decoder_out.output_tokens)] 
                    teacher_tgt=pad_sequence([tgt[tgt!=self.blank] for tgt in teacher_tgt], padding_value=self.pad, batch_first=True)
                    teacher_tgt=pad_sequence([tgt for tgt in teacher_tgt], padding_value=self.pad, batch_first=True)

                    if step_num == 0:
                        first_step_bleu = self.compute_bleu(teacher_tgt, tgt_tokens).score
                    elif step_num+1 == len(step_schedule):
                        second_step_bleu = self.compute_bleu(teacher_tgt, tgt_tokens).score
                        gap=second_step_bleu-first_step_bleu
                        print('bleu', gap, first_step_bleu, second_step_bleu)

            if self.args.teacher_beam_size>1:
                best_index = self.pick_best_bleu_index(prev_decoder_out.output_tokens, tgt_tokens)
                encoder_out, prev_decoder_out=self.index_tensors(encoder_out, prev_decoder_out, best_index)
                teacher_embds = teacher_embds[:, best_index]
                probs = probs[best_index]
                
            assert prev_decoder_out.output_tokens.shape[0]==len(prev_output_tokens)
            assert encoder_out['encoder_out'][0].shape[1]==len(prev_output_tokens)        
        return_dict={}
        if self.args.optimize_encoder:
            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
            
        if self.args.optimize_length_predictor:
            length_out = self.decoder.forward_length(
                normalize=False, encoder_out=encoder_out
            )
            length_tgt = self.decoder.forward_length_prediction(
                length_out, encoder_out, tgt_tokens
            )    
            return_dict["length"]={
                "out": length_out,
                "tgt": beam_length_tgt if self.args.beam_length else length_tgt,
                "factor": self.args.length_loss_factor,
            }
        
        word_ins_out, std_inner_states = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            inner_states=True,
        )
        std_embds, std_cross_attn, std_self_attn = std_inner_states['inner_states'], std_inner_states['cross-attn'], std_inner_states['self-attn']
        if self.args.revealed_loss:
            word_ins_mask &= prev_decoder_out.output_tokens.ne(self.unk)


        if self.args.oaxe_orig_loss_factor > 0.:
            sample={'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths, 'prev_output_tokens': prev_output_tokens.clone()}, "target": tgt_tokens}
            sample['net_input']['prev_output_tokens'][sample['net_input']['prev_output_tokens']!=self.pad] = self.unk
            word_ins_mask_oaxe = prev_output_tokens.eq(self.unk)
            word_ins_out_oaxe, _ = self.decoder(
                normalize=False,
                prev_output_tokens=sample['net_input']['prev_output_tokens'],
                encoder_out=encoder_out,
                inner_states=True,
            )
            net_output = self.generate_oaxe_net_output(word_ins_out_oaxe, word_ins_mask_oaxe, encoder_out, tgt_tokens)
            return_dict['oaxe_orig_loss']={'loss': self.oaxe(self, sample, net_output=net_output)[0], 'factor': self.args.oaxe_orig_loss_factor}

        if self.args.oaxe_distill_loss_factor>0:
            sample={'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths, 'prev_output_tokens': prev_output_tokens}, "target": prev_decoder_out.output_tokens}
            word_ins_out_oaxe, _ = self.decoder(
                normalize=False,
                prev_output_tokens=sample['net_input']['prev_output_tokens'],
                encoder_out=encoder_out,
                inner_states=True,
            )
            net_output = self.generate_oaxe_net_output(word_ins_out_oaxe, word_ins_mask, encoder_out, tgt_tokens)
            return_dict['oaxe_distill']={'loss': self.oaxe(self, sample, net_output=net_output)[0], 'factor': self.args.oaxe_distill_loss_factor}
        if self.args.distill_loss_factor>0:
            return_dict["word_ins"]={
                "out": word_ins_out,
                "tgt": teacher_tgt if getattr(self.args, 'criterion', 'nat_loss')=='ctc' else (probs if not self.args.hard_label else prev_decoder_out.output_tokens),  
                "mask": force_emits if getattr(self.args, 'criterion', 'nat_loss')=='ctc' else word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": 1./6. if self.args.layer_prediction_loss_factor>0 else self.args.distill_loss_factor,
            }
            if self.args.layer_prediction_loss_factor>0:
                self.per_layer_loss(std_embds, return_dict, teacher_tgt, word_ins_mask, force_emits)

       

        if self.args.correction_loss_factor > 0:
            correct_ins_out, _ = self.decoder(
                normalize=False,
                prev_output_tokens=corrupted_tokens,
                encoder_out=encoder_out,
                inner_states=True,
            )
            correction_mask=word_ins_mask&corrupted_tokens.ne(self.unk)
            return_dict["correction_loss"]={
                "out": correct_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": self.args.correction_loss_factor
            }

        def add_loss(name, factor, teacher, std, index):
            if factor > .0:
                loss = torch.norm(teacher - std, dim=-1, p=2)
                loss = loss[index].mean()
                return_dict[name] = {
                    "loss":  factor*loss,
                    "mask": torch.tensor(0),
                }

        for layer in range(self.args.embed_loss_layers):
            add_loss(f"embed-layer-{layer}", self.args.embed_loss_factor, teacher_embds[-(self.args.teacher_iterative_steps*layer+1)], std_embds[-(layer+1)], word_ins_mask.t())
        add_loss("cross-attn", self.args.cross_attn_loss_factor, teacher_cross_attn, std_cross_attn, word_ins_mask)
        add_loss("self-attn", self.args.self_attn_loss_factor, teacher_self_attn, std_self_attn, word_ins_mask)

        

        if self.args.orig_loss_factor > .0:
            return_dict["orig"]={
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor":self.args.orig_loss_factor,
            }
        if getattr(self.args, 'criterion', 'nat_loss')=='ctc':
            return return_dict, best_alignemnt
        else:
            return return_dict



@register_model_architecture("cmlm_distill", "cmlm_distill")
def cmlm_distill_base_architecture(args):
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
    args.teacher_path = getattr(args, "teacher_path", None)
