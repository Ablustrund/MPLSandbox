from typing import Tuple, List, Dict, Set, Union, Callable, Optional
import copy, logging, math
import torch
import torch.nn.functional as F
from deepspeed.runtime.zero import GatheredParameters
from transformers.deepspeed import is_deepspeed_zero3_enabled
from utils import synchronize_forward_on_stage3, synchronize_if_distributed

def top_p_logits(logits: torch.Tensor, topp: float=0., filter_value: float=0., min_topk: int=1) -> torch.Tensor:
    assert logits.dim() == 2
    trunc_logits = logits.clone()
    if topp > 0.:
        # logits : (batch_size, vocab_size)
        sprobs, sinds = torch.sort(logits, dim=-1, descending=True)
        mask = (sprobs.cumsum(dim=-1) - sprobs) >= topp
        mask[:, :min_topk] = False

        # Remove tokens with cumulative probability above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=sinds, src=mask)
        trunc_logits[mask] = filter_value
        trunc_logits.div_(trunc_logits.sum(dim=-1, keepdim=True))
        
    return trunc_logits

def get_banned_ngrams(prev_input_ids: torch.LongTensor, n=4, init_ngrams=None):
    generated_ngrams: List[Dict[Tuple[int, ...], Dict[int, int]]] = [{} for _ in range(prev_input_ids.size(0))] if init_ngrams is None else copy.deepcopy(init_ngrams)
    for idx, token_ids in enumerate(prev_input_ids.tolist()):
        ngrams = generated_ngrams[idx]
        for ngram in zip(*[token_ids[i:] for i in range(n)]):
            prefix = ngram[:-1]
            suffix = ngram[-1]
            # ngrams[prefix] = ngrams.get(prefix, set()) | set([suffix])
            suffixes = ngrams.setdefault(prefix, {})
            suffixes[suffix] = suffixes.get(suffix, 0) + 1
    return generated_ngrams

def get_blacklist_ngrams(blacklist_fname: str, tokenizer):
    blacklist_ngrams: Dict[int, Dict[Tuple[int, ...], Set[int]]] = {}
    with open(blacklist_fname, 'r') as f:
        for word in f:
            word = word.strip()
            ids: List[int] = tokenizer.txt2vec(word)
            if ids:
                prefix = tuple(ids[:-1])
                suffix = ids[-1]
                n = len(ids)
                if n not in blacklist_ngrams:
                    blacklist_ngrams[n] = {}
                blacklist_ngrams[n][prefix] = blacklist_ngrams[n].get(prefix, set()) | set([suffix])
        
    return blacklist_ngrams

def sort_value_index(values: torch.Tensor, indexes: torch.Tensor, descending: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    s_values, s_indexes = torch.sort(values, dim=-1, descending=descending)
    new_indexes = torch.gather(indexes, dim=-1, index=s_indexes)
    return s_values, new_indexes

class CustomizedGeneration:
    def __init__(self, model, debug=False) -> None:
        self.opt = model.opt
        self.NULL_IDX = model.NULL_IDX
        self.terminate_idxs = model.terminate_idxs
        
        self.reorder_encoder_states: Callable = model.reorder_encoder_states
        self.reorder_decoder_incremental_state: Callable = model.reorder_decoder_incremental_state
        self.forward: Callable = model.forward
        self._add_additional_decoding_params: Optional[Callable] = getattr(model, '_add_additional_decoding_params', None)
        self._rerank_beams: Optional[Callable] = getattr(model, '_rerank_beams', None)
        
        self.debug_mode = debug
        self.is_zero3: bool = is_deepspeed_zero3_enabled()
        self.model: torch.nn.Module = model
        
    def _cal_length_penalty(self, curr_length, beam_length_penalty):
        if self.opt.length_penalty_version == 'parlai':
            length_penalty = torch.pow(curr_length / 6., beam_length_penalty)
        elif self.opt.length_penalty_version == 'eva':
            length_penalty = torch.pow(curr_length, beam_length_penalty)
        else:
            raise ValueError
        
        return length_penalty
    
    def check_value_stability(self, logits):
        assert not torch.isnan(logits).any()
        assert not torch.isneginf(logits).any()
        assert not torch.isinf(logits).any()

    def _run_fake_forward(self):
        return self.model(torch.tensor([[0]], dtype=torch.long, device=self.model.device))
    
    @torch.no_grad()
    def generate(self, batch, **kwargs):
        if self.is_zero3:
            with GatheredParameters(self.model.parameters()):
                output = self._generate(batch, **kwargs)
            synchronize_forward_on_stage3(True, self._run_fake_forward)
            # synchronize_if_distributed()
            return output

        return self._generate(batch, **kwargs)
    
    @torch.no_grad()
    def _generate(self, batch, **kwargs):
        beam_size = kwargs.pop('beam_size', self.opt.beam_size)
        beam_groups = kwargs.pop('beam_groups', max(self.opt.beam_groups, 1))
        beam_group_delay = kwargs.pop('beam_group_delay', max(self.opt.group_delay, 1))
        tot_beam_size = beam_size * beam_groups
        
        max_ts = kwargs.pop('max_ts', self.opt.max_ts if self.opt.max_ts is not None else self.opt.label_truncate)
        init_temperature = kwargs.pop('init_temperature', self.opt.temperature)
        repetition_penalty = kwargs.pop('repetition_penalty', self.opt.repetition_penalty)
        beam_min_length = kwargs.pop('beam_min_length', self.opt.beam_min_length)
        inference = kwargs.pop('inference', self.opt.inference)
        beam_length_penalty = kwargs.pop('beam_length_penalty', self.opt.beam_length_penalty)
        topp = kwargs.pop('topp', self.opt.topp)
        ngram_no_repeat = kwargs.pop('ngram_no_repeat', self.opt.no_repeat_ngram)
        
        if inference in ('nucleus', 'beam') and (beam_groups != 1 or beam_group_delay != 1):
            logging.warn(f"nucleus sampling and pure beam search does NOT support beam_groups and beam_group_delay args. overrided by 1")
            beam_groups = beam_group_delay = 1

        decoder_input: torch.LongTensor = batch['text_vec']
        assert decoder_input[:, -1].ne(self.NULL_IDX).all(), 'For generation, last token should not be a padding token (you can use left padding instead)'
            
        dev = decoder_input.device
        bsz = decoder_input.size(0)
        _neg_inf = torch.tensor(-10000., dtype=torch.float16, device=dev)
        _zero = torch.tensor(1e-6, dtype=torch.float16, device=dev)

        # repeat encoder outputs and decoder inputs
        scores = torch.zeros((bsz * tot_beam_size, ), device=dev, dtype=torch.float16)
        done = torch.zeros((bsz * tot_beam_size, ), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, tot_beam_size).view(-1)
        additional_penalty_tokens = kwargs.get('penalty_tokens', None)
        # context_ids = self.reorder_encoder_states(decoder_input, inds)
        decoder_input = self.reorder_encoder_states(decoder_input, inds)
        additional_penalty_tokens = self.reorder_encoder_states(additional_penalty_tokens, inds) if additional_penalty_tokens != None else None
        init_length = decoder_input.size(1)
        
        # additional params
        other_params = {}
        if self._add_additional_decoding_params:
            other_params.update(self._add_additional_decoding_params(decoder_input, **kwargs))

        # Ban repeat ngrams
        if ngram_no_repeat > 1:
            context_banned_ngrams = get_banned_ngrams(decoder_input, n=ngram_no_repeat)
        else:
            context_banned_ngrams = [{} for _ in range(decoder_input.size(0))]
            
        incr_state = None
        for _ts in range(max_ts):
            if done.all():
                break
            # print(_ts)
            # score, incr_state, *_ = self.forward(decoder_input, incr_state, **other_params)
            score, *_ = self.forward(decoder_input, incr_state, **other_params)
            if self.is_zero3:
                # avoid deadlock on stage3
                synchronize_forward_on_stage3(False, None)
            score = score.half()

            # now score is bs*beam_size, len, vocab_size
            score = score[:, -1, :]
            
            # ----------------------
            # logits postprocessing
            # ----------------------
            
            # Exceed the max length, reduce temperature for quick end. no hard break
            if False:
                temperature = init_temperature * max(math.exp((max_ts - _ts) / 10.), 0.2)
            else:
                temperature = init_temperature
                
            # calculate repetition penalty
            if repetition_penalty > 1.:
                penalty_tokens = decoder_input[:, init_length:]
                if additional_penalty_tokens != None:
                    penalty_tokens = torch.cat((penalty_tokens, additional_penalty_tokens), dim=-1)
                penalty_scores = torch.gather(score, dim=1, index=penalty_tokens)
                penalty_scores = torch.where(penalty_scores < 0., penalty_scores * repetition_penalty, penalty_scores / repetition_penalty)
                score = score.scatter_(dim=1, index=penalty_tokens, src=penalty_scores)

            # Ban tokens
            # Ban repeat ngrams
            if _ts >= ngram_no_repeat - 1 and ngram_no_repeat > 1:
                banned_ngrams = get_banned_ngrams(decoder_input[:, init_length:], n=ngram_no_repeat, init_ngrams=context_banned_ngrams)
                prefixes: List[List[int]] = decoder_input[:, -ngram_no_repeat+1:].tolist()
                assert len(banned_ngrams) == len(prefixes) == score.size(0)
                for i, (ngrams, prefix) in enumerate(zip(banned_ngrams, prefixes)):
                    prefix = tuple(prefix)
                    for suffix, freq in ngrams.get(prefix, {}).items():
                        penalty = max(math.exp(freq / 10.), 1.5)
                        if score[i, suffix] > 0.:
                            score[i, suffix] /= penalty
                        else:
                            score[i, suffix] *= penalty
                        
            # TODO: Ban blacklist ngrams
            

            # Limit min length
            if _ts < beam_min_length:
                for end in self.terminate_idxs:
                    score[:, end] = _neg_inf

            # ----------------------
            # Decoding
            # ----------------------
            # perform sampling
            if inference == 'nucleus':
                # nucleus sampling
                score = torch.softmax(score.div(temperature), dim=-1)
                probs = top_p_logits(score, topp=topp, filter_value=0.)
                tok_ids = torch.multinomial(probs, 1)[:, 0]
                hyp_ids = torch.arange(probs.size(0), device=dev)
                scores = scores + probs[hyp_ids, tok_ids].log() * ~done

            # perform beamsearch
            elif inference in ('beam', 'beam_topp'):
                # The first step, we need to initialize topn candidates
                if _ts == 0:
                    score = score.view(bsz, tot_beam_size, -1)
                    score[:, 1:, :] = _neg_inf
                    score = score.view(bsz * tot_beam_size, -1)

                score = torch.log_softmax(score, dim=-1)
                score = torch.where(torch.isneginf(score), _neg_inf, score)
                # if self.debug_mode:
                #     # print(_ts)
                #     self.check_value_stability(score)
                voc_size = score.size(-1)
                beam_scores = scores.unsqueeze(-1) + score
                
                # never sample tokens from beams already done
                beam_scores[done, :] = _neg_inf
                
                if _ts < beam_group_delay:
                    beam_scores = beam_scores.view(bsz, -1) # bsz, group_size * beam_size * vocab_size
                else:
                    beam_scores = beam_scores.view(bsz * beam_groups, -1) # bsz * group_size, beam_size * vocab_size
                
                # pad tokens for done samples
                if _ts < beam_group_delay:
                    pad_idx_offset = torch.arange(tot_beam_size, device=dev, dtype=torch.long).unsqueeze(0).repeat(bsz, 1) * voc_size + self.NULL_IDX
                    done_beams = done.view(bsz, tot_beam_size)
                else:
                    pad_idx_offset = torch.arange(beam_size, device=dev, dtype=torch.long).unsqueeze(0).repeat(bsz * beam_groups, 1) * voc_size + self.NULL_IDX
                    done_beams = done.view(bsz * beam_groups, beam_size)
                n_done_beams = done_beams.sum(-1)
                
                num_to_sample = tot_beam_size if _ts < beam_group_delay else beam_size
                if inference == 'beam_topp' and _ts > 0:
                    # perform sampling on beam search
                    beam_scores.div_(temperature)
                    beam_probs = beam_scores.softmax(dim=-1) # softmax again to get probs
                    beam_probs = top_p_logits(beam_probs, topp=topp, filter_value=0., min_topk=num_to_sample)
                    
                    best_idxs = torch.multinomial(beam_probs, num_to_sample)
                    best_scores = torch.gather(beam_probs, dim=-1, index=best_idxs)
                    best_scores, best_idxs = sort_value_index(best_scores, best_idxs, descending=True)
                    
                    pad_score = 1.
                else:
                    best_scores, best_idxs = torch.topk(beam_scores, num_to_sample, dim=-1, sorted=True)
                    pad_score = 0.
                    
                # best_scores, best_idxs = best_scores[:, :num_to_sample], best_idxs[:, :num_to_sample]
                for i in range(pad_idx_offset.size(0)):
                    if n_done_beams[i] > 0:
                        best_scores[i, -n_done_beams[i]:] = pad_score
                        best_idxs[i, -n_done_beams[i]:] = pad_idx_offset[i, done_beams[i]]
                    
                # get the selected token score
                score = torch.gather(score.view(best_idxs.size(0), -1), dim=-1, index=best_idxs).view(-1)
                if _ts < beam_group_delay:
                    # here we treat all groups as one so 
                    # the size is bsz, group_size * beam_size, resize is needed
                    best_scores = best_scores.view(bsz * beam_groups, beam_size)
                    best_idxs = best_idxs.view(bsz * beam_groups, beam_size)
                # get the backtracking hypothesis id as a multiple of full voc_sizes
                hyp_ids = torch.div(best_idxs, voc_size, rounding_mode='trunc')  # bsz, beam_size
                # get the actual word id from residual of the same division
                tok_ids = (best_idxs % voc_size).view(-1)

                # select corresponding samples
                if _ts < beam_group_delay:
                    # now the hyp_ids is related to bsz
                    hyp_offset = torch.arange(0, bsz).to(dev).unsqueeze(-1).repeat(1, beam_groups).view(bsz * beam_groups, 1) * tot_beam_size
                else:
                    # now the hyp_ids is related to bsz * beam_groups
                    hyp_offset = torch.arange(0, bsz * beam_groups).to(dev).unsqueeze(-1) * beam_size  # bsz, 1
                hyp_ids = (hyp_ids + hyp_offset).view(-1)

                decoder_input = torch.index_select(decoder_input, dim=0, index=hyp_ids)
                scores = torch.index_select(scores, dim=0, index=hyp_ids)
                done = torch.index_select(done, dim=0, index=hyp_ids)

                scores = scores + score * ~done
            
            else:
                raise ValueError
            # print(tok_ids)
            # print(hyp_ids)
            # print(score)
            # print(best_scores)
            tok_ids = torch.where(done, self.NULL_IDX, tok_ids)
            decoder_input = torch.cat((decoder_input, tok_ids.unsqueeze(-1)), dim=-1)
            for end in self.terminate_idxs:
                done = done | tok_ids.eq(end)
            # incr_state = self.reorder_decoder_incremental_state(incr_state, hyp_ids)
            # print(decoder_input)
            # print(scores)
            # print('-----------')
            # time.sleep(1)
            # assert not tok_ids[~done].eq(self.NULL_IDX).any(), f"{tok_ids}, {score}, {best_scores}"

        # get all finalized candidates for each sample
        decoder_input = decoder_input[:, init_length:]
        decoder_input = decoder_input.view(bsz, beam_groups, beam_size, -1)
        scores = scores.view(bsz, beam_groups, beam_size)
        # print(scores)
        lengths = decoder_input.ne(self.NULL_IDX).sum(dim=-1)
        # print(lengths)
        
        length_penalty = self._cal_length_penalty(lengths, beam_length_penalty)
        scores /= length_penalty

        # print(scores)
        if self._rerank_beams:
            scores = self._rerank_beams(
                batch, decoder_input.view(bsz * tot_beam_size, decoder_input.size(-1)), scores
            )

        n_best_beam_preds_scores: List[List[Tuple[float, List[int]]]] = []
        for i in range(bsz):
            beams: List[Tuple[float, List[int]]] = []
            for k in range(beam_groups):
                group_beams: List[Tuple[float, List[int]]] = []
                for j in range(beam_size):
                    seq: torch.LongTensor = decoder_input[i, k, j, :lengths[i, k, j]]
                    # assert seq[-1] == self.END_IDX, f'illegal generated sequence {decoder_input[i, k, j]}, {i}, {k}, {j}, {scores[i, k, j]}'
                    # if seq[-1].item() in self.terminate_idxs:
                    group_beams.append((float(scores[i, k, j]), seq.tolist()))
                    # else:
                        # group_beams.append((-99. + float(scores[i, k, j]), seq.tolist()))
                group_beams.sort(reverse=True, key=lambda x: x[0])
                if beam_groups > 1:
                    beams.append(group_beams[0])
                else:
                    beams.extend(group_beams)
            beams.sort(reverse=True, key=lambda x: x[0])
            n_best_beam_preds_scores.append(beams)

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, n_best_beam_preds_scores


if __name__ == '__main__':
    # dummy_input = torch.randn((3, 4)).softmax(dim=-1)
    # print(dummy_input)
    # print(top_p_logits(dummy_input, topp=0.6))
    
    dummy_ids = torch.LongTensor(
        [[1, 2, 3, 4, 5, 1, 2, 3, 5, 1, 2, 3, 5],
         [2, 3, 4, 4, 4, 2, 3, 4, 8, 2, 3, 4, 8],
         [3, 4, 5, 5, 6, 4, 5, 5, 7, 4, 5, 5, 7]])
    print(get_banned_ngrams(dummy_ids))