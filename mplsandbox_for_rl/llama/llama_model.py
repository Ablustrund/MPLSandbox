import torch
import math, time, logging
from typing import Dict, Any, Union, List, Tuple
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from tokenizer import HFPretrainedTokenizer as LlamaPretrainedTokenizer
from utils import build_prompt, get_separate_prompt, cut_uncomplete_sentence
from generate_utils import CustomizedGeneration


class Llama(LlamaForCausalLM):
    def __init__(self, config, opt: Dict[str, Any], dict: LlamaPretrainedTokenizer, **kwargs):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = dict
        
        self.debug_mode: bool = opt.debug
        
        self.end_idx = dict.end_token_id
        self.NULL_IDX = dict.null_token_id
        self.terminate_idxs = [self.end_idx]
        
        if opt.use_huggingface_generate:
            self.forward = super().forward
            
        self.post_init()
        
    def forward(self, decoder_input: torch.LongTensor, incr_state: torch.Tensor=None):
        # print(decoder_input.dtype)
        attention_mask = decoder_input.ne(self.NULL_IDX)
            
        if incr_state is not None:
            decoder_input = decoder_input[:, -1:]
            
        output = super().forward(input_ids=decoder_input, 
                                 past_key_values=incr_state, 
                                 attention_mask=attention_mask, return_dict=True, use_cache=not self.training)
        logits = output.logits
        new_incr_states = output.past_key_values
        # pred = logits.argmax(-1)
        
        return logits, new_incr_states
        
    def reorder_encoder_states(self, encoder_states, indices):
        enc = torch.index_select(encoder_states, 0, indices)
        return enc
    
    def reorder_decoder_incremental_state(self, incremental_state, inds):
        return self._reorder_cache(incremental_state, inds)
        
    def _preprocess_context(self, context: Union[str, List[str]], **kwargs):
        if isinstance(context, str):
            assert self.opt.no_prompt
            # context = context.strip().split('\n')
            
        # additional info
        knowledge = kwargs.get('knowledge', None)
        init_role, inline_role = kwargs.get('role', (None, None))
        
        # context
        if isinstance(context, list) and not self.opt.multi_role:
            context = [get_separate_prompt(i + (len(context) + 1) % 2) + u for i, u in enumerate(context)]
        context_maxl = self.opt.context_truncate - self.opt.max_ts
        
        build_prompt_kwargs = {'openai_style': self.opt.openai_style_prompt} # args to build different prompt
        # preprocess knowledge
        if knowledge is not None:
            assert init_role is None and inline_role is None
            kd_vec = self.tokenizer.txt2vec(knowledge + '\n\n')
        else:
            kd_vec = []
            
        # preprocess role info
        if init_role is not None and inline_role is not None:
            assert knowledge is None
            del build_prompt_kwargs['openai_style']
            build_prompt_kwargs.update({'init_prompt': init_role, 'inline_prompt': inline_role, 'merge': self.opt.merge_role_prompts})
            
        if self.opt.no_split_dialog:
            build_prompt_kwargs['dialog_sep'] = self.tokenizer.end_token
        else:
            build_prompt_kwargs['dialog_sep'] = self.opt.delimiter
            
        # encode and truncate
        while True:
            context_str = build_prompt(context, **build_prompt_kwargs) if isinstance(context, list) else context
            context_vec = self.tokenizer.txt2vec(context_str)
            if isinstance(context, str):
                # non dialog data, truncate directly
                context_vec = context_vec[-context_maxl:]
            if len(kd_vec) + len(context_vec) <= context_maxl or len(context) == 1:
                break
            else:
                # is dialog data, truncate oldest dialog history
                assert isinstance(context, list)
                context = context[1:]
                
        # concate all
        context_vec = kd_vec + context_vec

        if self.debug_mode:
            print(f'Context: \n{self.tokenizer.vec2txt(context_vec)}')
        return context_vec
    
    def generate_sentence(self, context: Union[str, List[str]], **kwargs):
        context_vec = self._preprocess_context(context, **kwargs)
        context_vec = torch.tensor([context_vec], dtype=torch.long, device='cuda')
        inputs = {
            'text_vec': context_vec
        }
        outputs, candids = self.generate(inputs)
        
        score, resp_ids = outputs[0]
        resp = self.tokenizer.vec2txt(resp_ids, skip_special=True).strip()
        
        # handle sentence that exceed max_ts and not completely generated
        if resp_ids[-1] not in self.terminate_idxs:
            resp = cut_uncomplete_sentence(resp)
        
        candids = '\n'.join(self.tokenizer.vec2txt(candid, skip_special=True).strip() for _, candid in candids[0])
        return score, resp, candids
    
    @torch.no_grad()
    def generate(self, batch: Dict[str, Any], **kwargs):
        beam_size = self.opt.beam_size
        max_ts = self.opt.max_ts if self.opt.max_ts is not None else self.opt.label_truncate
        init_temperature = self.opt.temperature
        repetition_penalty = self.opt.repetition_penalty
        beam_min_length = self.opt.beam_min_length
        inference = self.opt.inference
        beam_length_penalty = self.opt.beam_length_penalty
        topp = self.opt.topp
        
        # ======================== use huggingface generate pipeline ==================
        if self.opt.use_huggingface_generate:
            # forward method backup and override
            forward_fn_backup = self.forward
            self.forward = super().forward
            
            inputs = batch['text_vec']
            bsz, init_length = inputs.size(0), inputs.size(1)
            outputs = super().generate(inputs, 
                                       max_new_tokens=max_ts, 
                                       num_return_sequences=beam_size if inference != 'nucleus' else 1,
                                       num_beams=beam_size if inference != 'nucleus' else 1, 
                                       top_p=topp, 
                                       repetition_penalty=repetition_penalty, 
                                       do_sample=inference != 'beam', 
                                       min_new_tokens=beam_min_length, 
                                       length_penalty=beam_length_penalty,
                                       temperature=init_temperature,
                                       output_scores=True,
                                       return_dict_in_generate=True)
            seqs = outputs.sequences.view(bsz, beam_size, -1).tolist()
            scores = getattr(outputs, 'sequences_scores', torch.tensor([[0.] * beam_size for _ in range(bsz)]))
            scores = scores.view(bsz, 1).tolist()
            # if self.debug_mode:
            #     print(inputs)
            #     print(outputs)
            n_best_beam_preds_scores = [[(score, candid[init_length:]) for score, candid in zip(scores[i], seqs[i])] for i in range(bsz)]
            beam_preds_scores = [o[0] for o in n_best_beam_preds_scores]
            # forward method restore
            self.forward = forward_fn_backup

            return beam_preds_scores, n_best_beam_preds_scores
            
                    
        # =========================== use customized generate pipeline =========================
        generate_fn = CustomizedGeneration(self, self.debug_mode)

        return generate_fn.generate(batch, **kwargs)



