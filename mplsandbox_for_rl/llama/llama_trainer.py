import torch
import time
from typing import Dict, Any, Tuple
import torch.nn as nn
from .llama_model import Llama
from data_helper import *
from trainer import Seq2SeqTrainer
from tokenizer import HFPretrainedTokenizer as LlamaPretrainedTokenizer
from metric import MeanMetric, RealtimeMetric, SumMetric
from ppo.ppo_datahelper import *


class LlamaTrainer(Seq2SeqTrainer):
    model: Llama
    def __init__(self, opt, model: Llama, reward_model: nn.Module, accelerator: Accelerator, eval_only=False) -> None:
        super().__init__(opt, model, accelerator, eval_only)
        self.add_kd: bool = opt.add_kd
        self.reward_model = reward_model
    
    def _criterion(self, model_output: Tuple[torch.Tensor, ...], batch: Dict[str, Any], return_output=False, training=True):
        scores, *_ = model_output
        scores = scores[:, :-1, :]
        preds = scores.argmax(dim=-1)
        
        label_vec = batch['text_vec'][:, 1:].clone()
        loss_mask = batch['loss_mask'][:, 1:]
        label_vec[~loss_mask] = self.tokenizer.null_token_id
        batch['label_vec'] = label_vec
        return super()._criterion((scores, preds), batch, return_output, training)
        
    def _tokenizer_class(self):
        return LlamaPretrainedTokenizer

    def _dataset_class(self):
        # if self.opt.add_kd:
        #     # from ..bloom.knowledge.kd_data_helper import BloomKnowledgeDataset
        #     from .knowledge.kd_data_helper import BloomKnowledgeDataset
        #     return BloomKnowledgeDataset
        # if self.opt.add_role:
        #     from bloom.role.role_data_helper import BloomRoleDataset
        #     return BloomRoleDataset
        # if self.opt.no_prompt:
        #     from bloom.pretrain.pretrain_data_helper import PretrainChunkDataset, PretrainDataset
        #     if self.opt.use_chunk_data:
        #         return PretrainChunkDataset
        #     else:
        #         return PretrainDataset
        return super()._dataset_class()
    
    def _prompt_dataset_class(self):
        return DialogPromptDataset
    
    def _strip_pad(self, seq: List[int]):
        return [tok for tok in seq if tok != self.tokenizer.null_token_id]

    def _extract_context_candidates(self, context: List[List[int]], candidates: List[List[Tuple[float, List[int]]]]):
        assert len(context) == len(candidates), f'Batchsize not match {len(context)} & {len(candidates)}'
        all_context, all_resp = [], []
        for ctx, candidate in zip(context, candidates):
            ctx = self._strip_pad(ctx)
            for _, resp in candidate:
                resp = self._strip_pad(resp)
                # print(resp)
                if resp[-1] != self.tokenizer.end_token_id:
                    logging.warn(f'Found too long generated response')
                all_context.append(ctx.copy())
                all_resp.append(resp)
                
        all_context_resp = [c + r for c, r in zip(all_context, all_resp)]
        return all_context, all_resp, all_context_resp
    
    def _build_metrics(self, mode='train'):
        metrics = super()._build_metrics(mode)

        if mode == 'eval_skip_generation':
            metrics.add_additional_metric('rewards', MeanMetric())
            metrics.add_additional_metric('r1ward_num', MeanMetric())
            metrics.add_additional_metric('introductory', SumMetric())
            metrics.add_additional_metric('interview', SumMetric())
            metrics.add_additional_metric('competition', SumMetric())
        return metrics
    
    @torch.no_grad()
    def evaluate(self, datatype='valid', **kwargs) -> Tuple[float, List]:
        assert datatype in ('valid', 'test')
        start_time = time.time()
        n_generated = 0
        valid_dataloader = get_dataloader(self._prompt_dataset_class()(self.opt, self.accelerator, mode=datatype), self.opt)
        write_log_info_on_rank0(f'Start evaluation on {datatype} data')
        self.model.eval()
        
        for step, batch in enumerate(valid_dataloader):
            if n_generated >= self.opt.num_examples:
                break
            # record some info
            self._record_batch_info(batch, mode='valid')
            to_cuda(batch)
            outputs, candidates = self.model.generate(batch, beam_size=1, temperature=0.2, top_p=0.95, max_ts=1024, **kwargs)
            _, resp_vec, output_vec = self._extract_context_candidates(batch['text_vec'].tolist(), candidates)
            # if self.debug:
            #     logging.info(self.tokenizer.vec2txt(output_vec[-1]))
            
            output_vec = torch.tensor(pad_sequences(output_vec, pad_value=self.tokenizer.null_token_id, pad_left=True), 
                                                     dtype=torch.long, device=self.accelerator.device)
            
            bsz = output_vec.size(0)
            resps = [self.tokenizer.vec2txt(resp_vec[i]) for i in range(bsz)]
            rewards, pass_list, introductory, interview, competition = self.reward_model.forward(resp_vec, resps, batch, bsz, mode='valid')
            
            rewards = rewards.tolist()
            # print(f'------{rewards}------')
            assert len(rewards) == output_vec.size(0), f"{rewards.size()}, {output_vec.size()}"
            self.valid_metrics.record_metric_many('rewards', rewards)
            self.valid_metrics.record_metric_many('r1ward_num', pass_list)
            self.valid_metrics.record_metric_many('introductory', introductory)
            self.valid_metrics.record_metric_many('interview', interview)
            self.valid_metrics.record_metric_many('competition', competition)
            
            n_generated += len(candidates)
            
        # log info
        metrics = self.valid_metrics.all_gather_metrics()
        self.valid_metrics.display(self.custom_states.total_steps, gathered_metrics=metrics)
        self.valid_metrics.write_tensorboard(self.custom_states.total_steps, gathered_metrics=metrics)
        self.valid_metrics.flush()
        validation_score = metrics[self.validation_metric]
        self.valid_metrics.reset(no_reset=[])
        
        write_log_info_on_rank0(f'Evaluation completed in {(time.time() - start_time):.2f} seconds')
        self.model.train()
        torch.cuda.empty_cache()
        return validation_score, None
 

    @torch.no_grad()
    def inference(self, datatype, reward_model, **kwargs) -> Tuple[float, List]:
        # assert datatype in ('valid', 'test')
        start_time = time.time()
        n_generated = 0
        test_dataloader = get_dataloader(self._prompt_dataset_class()(self.opt, self.accelerator, mode=datatype), self.opt)
        print(f'Start evaluation on {datatype} data')
        self.model.eval()

        from collections import defaultdict
        cate_count = defaultdict(int)
        from tqdm import tqdm
        for batch in tqdm(test_dataloader):
            if n_generated >= self.opt.num_examples:
                break
            # record some info
            self._record_batch_info(batch, mode='valid')
            to_cuda(batch)
            # print(batch)
            try:
                # print('begin')
                print(batch["text_vec"])
                print(batch["text"])
                outputs, candidates = self.model.generate(batch, beam_size=1, **kwargs)
                # print(candidates)
            except Exception as e:
                print(f"exception happended ：{e}")
                continue
            context_vec_sampled, resp_vec_sampled, output_vec = self._extract_context_candidates(batch['text_vec'].tolist(), candidates)
            bsz = self.opt.rollout_batch_size
            # calculate & normalize reward
            # print(resp_vec_sampled)
            resps = [self.tokenizer.vec2txt(resp_vec_sampled[i]) for i in range(bsz)]
            # print(resps)
            rewards, pass_list = self.reward_model.forward(resp_vec_sampled, resps, batch, bsz, mode='test', **kwargs)

            for i in range(bsz):
                if pass_list[i] == 1:
                    cate_count[batch[i]["difficulty"]] += 1
                
        cate_count["introductory"] = round(cate_count["introductory"]/97, 2)
        cate_count["interview"] = round(cate_count["interview"]/99, 2)
        cate_count["competition"] = round(cate_count["competition"]/44, 2)

        print(f'Evaluation completed in {(time.time() - start_time):.2f} seconds')