# Add grad to beam search

import math

import torch

from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.search import Search


class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        stop_early=True,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            stop_early (bool, optional): stop generation immediately after we
                finalize beam_size hypotheses, even though longer hypotheses
                might have better normalized scores (default: True)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = BeamSearch(tgt_dict)

    def generate(
        self,
        models,
        sample,
        prefix_tokens=None,
        bos_token=None,
        noise = None,# maple
        prior = None, # maple
        **kwargs
    ):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
        """
        training_state = models[0].training
        model = EnsembleModel(models)
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        #scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores = src_tokens.new(bsz * beam_size, 0).float().fill_(0)
        scores_buf = scores.clone()
        full_scores = src_tokens.new(bsz * beam_size, 0, self.vocab_size).float().fill_(0)# maple
        #tokens = src_tokens.data.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens = src_tokens.data.new(bsz * beam_size, 1).long().fill_(self.pad)

        tokens_buf = tokens.clone()
        tokens[:, 0] = bos_token or self.eos
        attn, attn_buf = None, None
        nonpad_idxs = None
        if prefix_tokens is not None:
            partial_prefix_mask_buf = torch.zeros_like(src_lengths).byte()

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == max_len or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[unfin_idx].max()
                if self.normalize_scores:
                    best_unfinalized_score /= max_len ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None, lprobs = None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            #tokens_clone[:, step] = self.eos #maple
            tokens_clone = torch.cat([tokens_clone, tokens_clone.new_full((tokens_clone.shape[0], 1), self.eos)], 1)
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            #pos_scores[:, step] = eos_scores #maple
            pos_scores = torch.cat([pos_scores, eos_scores.view(eos_scores.shape[0], 1)], 1)

            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            #maple
            pos_full_scores = full_scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_full_scores = torch.cat([pos_full_scores, lprobs[bbsz_idx].unsqueeze(1)], 1)

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            #for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())): #maple
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores)):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                        'full_scores': pos_full_scores[i],#maple
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        p_penalty = scores.new_zeros(1, len(models[0].decoder.dictionary.indices))
        p_penalty[0, self.pad] = math.inf
        p_penalty[0, self.unk] = self.unk_penalty
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    #reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    reorder_state = (reorder_state.view(-1, beam_size) + (corr.unsqueeze(-1) * beam_size)).view(-1)
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)
            #print(tokens.shape, encoder_outs[0]['encoder_out'].shape)
            lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
            )

            #lprobs[:, self.pad] = lprobs[:, self.pad] - math.inf  # never select pad # maple
            #lprobs[:, self.unk] = lprobs[:, self.unk] - self.unk_penalty  # apply unk penalty #maple
            lprobs = lprobs - p_penalty

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < max_len:
                self.search.set_src_lengths(src_lengths)

                if self.no_repeat_ngram_size > 0:
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, prevent decoding of ngrams that have already appeared
                        ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                        return gen_ngrams[bbsz_idx].get(ngram_index, [])

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                    else:
                        banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                    for bbsz_idx in range(bsz * beam_size):
                        lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    assert isinstance(self.search, search.BeamSearch) or bsz == 1, \
                            "currently only BeamSearch supports decoding with prefix_tokens"
                    probs_slice = lprobs.view(bsz, -1, lprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1)
                    ).view(-1, 1).repeat(1, cand_size)
                    if step > 0:
                        # save cumulative scores for each hypothesis
                        cand_scores.add_(scores[:, step - 1].view(bsz, beam_size).repeat(1, 2))
                    cand_indices = prefix_tokens[:, step].view(-1, 1).repeat(1, cand_size)
                    cand_beams = torch.zeros_like(cand_indices)

                # handle prefixes of different lengths
                # when step == prefix_tokens.size(1), we'll have new free-decoding batches
                if prefix_tokens is not None and step <= prefix_tokens.size(1):
                    if step < prefix_tokens.size(1):
                        partial_prefix_mask = prefix_tokens[:, step].eq(self.pad)
                    else:   #  all prefixes finished force-decoding
                        partial_prefix_mask = torch.ones(bsz).to(prefix_tokens).byte()
                    if partial_prefix_mask.any():
                        # track new free-decoding batches, at whose very first step
                        # only use the first beam to eliminate repeats
                        prefix_step0_mask = partial_prefix_mask ^ partial_prefix_mask_buf
                        lprobs.view(bsz, beam_size, -1)[prefix_step0_mask, 1:] = -math.inf
                        partial_scores, partial_indices, partial_beams = self.search.step(
                            step,
                            lprobs.view(bsz, -1, self.vocab_size),
                            scores.view(bsz, beam_size, -1)[:, :, :step],
                            prior = prior, # maple
                        )
                        cand_scores[partial_prefix_mask] = partial_scores[partial_prefix_mask]
                        cand_indices[partial_prefix_mask] = partial_indices[partial_prefix_mask]
                        cand_beams[partial_prefix_mask] = partial_beams[partial_prefix_mask]
                        partial_prefix_mask_buf = partial_prefix_mask

                else:
                    cand_scores, cand_indices, cand_beams = self.search.step(
                        step,
                        lprobs.clone().view(bsz, -1, self.vocab_size),# maple + clone
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                        noise, #maple
                        prior = prior, # maple
                    )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit max_len
                # pick the hypothesis with the highest prob of EOS right now
                eos_scores, eos_bbsz_idx = torch.sort(
                    lprobs[:, self.eos], #maple
                    descending=True,
                    #out=(eos_scores, eos_bbsz_idx),
                )# maple
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores, lprobs = lprobs))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.min_len:
                # only consider eos when it's among the top beam_size indices
                eos_bbsz_idx = torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    #out=eos_bbsz_idx,
                )# maple
                if eos_bbsz_idx.numel() > 0:
                    eos_scores = torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        #out=eos_scores,
                    ) #maple
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores, lprobs = lprobs)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                    partial_prefix_mask_buf = partial_prefix_mask_buf[batch_idxs]
                src_lengths = src_lengths[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                #full_scores = full_scores[batch_idxs]#maple
                full_scores = full_scores.view(bsz, -1, full_scores.shape[2])[batch_idxs].view(new_bsz * beam_size, -1, full_scores.shape[2]) # maple
                #lprobs = lprobs[batch_idxs]#maple
                lprobs = lprobs.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1) # maple
                #scores_buf.resize_as_(scores) # maple
                scores_buf = torch.empty_like(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                #tokens_buf.resize_as_(tokens) # maple
                tokens_buf = torch.empty_like(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    # attn_buf.resize_as_(attn) #maple
                    attn_buf = torch.empty_like(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            # active_mask = buffer('active_mask') maple
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                #out=active_mask,
            )#maple

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            #active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore') #maple
            _ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                #out=(_ignore, active_hypos)
            )#maple

            #active_bbsz_idx = buffer('active_bbsz_idx') # maple
            active_bbsz_idx = torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                #out=active_bbsz_idx,
            ) #maple
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                #out=scores[:, step].view(bsz, beam_size),
            )
            scores = torch.cat([scores, active_scores.view(-1, 1)], 1)# maple
            full_scores = torch.cat([full_scores, lprobs.unsqueeze(1)], 1)# maple

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            tokens_buf = torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                #out=tokens_buf[:, :step + 1],
            )# maple
            tmp = torch.gather(
                cand_indices, dim=1, index=active_hypos,
                #out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            ) # maple
            tokens_buf = torch.cat([tokens_buf, tmp.view(-1, 1)], 1)
            if step > 0:
                scores_buf = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    #out=scores_buf[:, :step],
                ) #maple
            tmp = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                #output = scores_buf.view(bsz, beam_size, -1)[:, :, step]
            ) # maple
            scores_buf = torch.cat([scores_buf, tmp.view(-1, 1)], 1)

            if prior is not None:#maple
                prior = torch.index_select(
                        prior, dim=0, index=active_bbsz_idx,
                    ) 

            # copy attention for active hypotheses
            if attn is not None:
                attn_buf[:, :, :step + 2] = torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    #out=attn_buf[:, :, :step + 2],
                ) #maple

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        if training_state:
            model.train()

        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.decoder(tokens, encoder_out, incremental_state=self.incremental_states[model]))
        else:
            decoder_out = list(model.decoder(tokens, encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        if attn is not None:
            if type(attn) is dict:
                attn = attn['attn']
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


# from search.py
class BeamSearch(Search):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def step(self, step, lprobs, scores, noise = None, prior = None):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        if noise is not None:
            ori_lprobs = lprobs
            lprobs = ori_lprobs + noise * torch.randn_like(lprobs)

        if prior is not None:
            lprobs = lprobs + prior.unsqueeze(1) * 0.2

        self.scores_buf, self.indices_buf = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            )
        ) # maple
        if noise is not None:
            self.scores_buf = ori_lprobs.view(bsz, -1).gather(1, self.indices_buf)
        #torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.beams_buf = torch.div(self.indices_buf, vocab_size)
        #self.indices_buf.fmod_(vocab_size)
        self.indices_buf = self.indices_buf.fmod(vocab_size)

        return self.scores_buf, self.indices_buf, self.beams_buf
