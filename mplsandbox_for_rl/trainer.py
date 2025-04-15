import logging
import time, math, json
from typing import Callable, Dict, Any, Union, Callable, Optional, Tuple, List
import torch
import torch.optim as optim
import torch.nn as nn
import subprocess
import _thread
from accelerate import Accelerator
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers.deepspeed import is_deepspeed_zero3_enabled

from data_helper import *
from metric import Metrics as Seq2SeqMetrics
from utils import *
from scheduler import *


class CustomTrainerStates:
    def __init__(self) -> None:
        self.total_steps = 0
        self.total_exps = 0
        self.patient = 0
        self.best_score = -9999999999
    
    def state_dict(self):
        return {
            'total_steps': self.total_steps,
            'total_exps': self.total_exps,
            'patient': self.patient,
            'best_score': self.best_score,
        }
        
    def load_state_dict(self, state_dict):
        self.total_steps = state_dict['total_steps']
        self.total_exps = state_dict['total_exps']
        self.patient = state_dict['patient']
        self.best_score = state_dict['best_score']
    

class Seq2SeqTrainer():
    def __init__(self, opt, model: nn.Module, accelerator: Accelerator, eval_only=False) -> None:
        self.opt = opt
        self.eval_only = eval_only
        self.no_reset_metric_names = ['total_exs'] # metrics won't be reset for every 50 steps
        self.print_interval = 25
            
        self.model = model
        self.accelerator = accelerator
        if not eval_only:
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            self.train_loader = self._build_dataloader()
            self.train_metrics = self._build_metrics('train')
            self.train_size = len(self.train_loader.dataset)
            
        self.tokenizer = self._tokenizer_class()(opt)
        # self.detokenizer = MosesDetokenizer(lang=opt['lang'])
        self.valid_metrics = self._build_metrics('eval_skip_generation' if opt.skip_generation else 'eval')
        self.loss_fn = self._build_loss_fn()
        
        self.custom_states = self._custom_states_class()()
        # self.grad_check_again = True
        self.max_steps: int = opt.train_steps
        # self.max_grad_norm = opt.grad_norm
        self.save_freq = opt.save_freq
        self.skip_generation_on_eval: bool = opt.skip_generation
        self.save_path = opt.model_file
        self.validation_metric = opt.validation_metric
        self.fp32_loss: bool = opt.fp32_loss
        assert self.validation_metric in getattr(self, 'train_metrics', self.valid_metrics).metrics, '--validation_metric is not specified in metrics'
        
        # load initial model
        if self.opt.init_model and os.path.exists(self.opt.init_model) and not os.path.isdir(self.opt.init_model):
            self._load_checkpoint(self.opt.init_model, strict=False)
            
        # prepare all things, DO **NOT** prepare dataloader
        self.accelerator.register_for_checkpointing(self.custom_states)
        if not self.eval_only:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        
        # load checkpoint
        if self.opt.model_file and os.path.exists(self.opt.model_file):
            self._load_checkpoint(self.opt.model_file, strict=True)
        
        # inference speedup
        # if self.eval_only:
        #     self.inference_engine = deepspeed.init_inference(self.model, mp_size=1, dtype=torch.half, replace_with_kernel_inject=True)
        #     self.model = self.inference_engine.module
        
        # For subclass extention: more initialization procedures before sync params
        self.post_init()
        
        synchronize_if_distributed()
            
    def post_init(self):
        pass
            
    def _load_checkpoint(self, load_path, strict=True, target_model: nn.Module = None):        
        # check if exist
        if load_path is None or not os.path.exists(load_path):
            raise FileNotFoundError

        synchronize_if_distributed()
        if os.path.isdir(load_path):
            # load all trainer status
            assert target_model is None
            if self.eval_only:
                return
            self.accelerator.load_state(load_path)
            self.train_metrics.record_metric('total_exs', self.custom_states.total_exps)
            return
        else:
            # load model only
            write_log_info_on_rank0(f'Load existing model from {load_path}')
            ckpt = torch.load(load_path, map_location='cpu')
            try:
                state_dict = ckpt['model']
            except KeyError:
                state_dict = ckpt
            
            if target_model is None:
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)
            else:
                missing_keys, unexpected_keys = target_model.load_state_dict(state_dict, strict=strict)
            
            if missing_keys or unexpected_keys:
                write_log_info_on_rank0(f"Missing weights: {missing_keys}\nUnexpected weights: {unexpected_keys}")
        
    def _save_checkpoint(self, is_best: bool, total_steps: int):
        best_model_path = os.path.join(self.save_path, 'best_model')
        steps_model_path = os.path.join(self.save_path, '{}_steps'.format(total_steps))
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if is_best:
            # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
            # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
            # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
            # For Zero Stages 1 and 2, models are saved as usual in the output directory.
            # The model name saved is `pytorch_model.bin`
            unwrapped_model.save_pretrained(
                best_model_path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(self.model),
            )
            logging.info(f'Saved best model to {best_model_path}')
        
        unwrapped_model.save_pretrained(
            steps_model_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
        )
        logging.info(f'Saved model of {total_steps} steps to {steps_model_path}')

        synchronize_if_distributed()
        # # save all training states
        self.accelerator.save_state(self.save_path)
        # upload checkpoints to hdfs
        # self._upload_checkpoint(is_best=is_best, n_steps=total_steps)
    
    def _dataset_class(self):
        if self.opt.use_chunk_data:
            return ChunkDataset
        else:
            return CodeDataset
    
    def _tokenizer_class(self):
        return HFPretrainedTokenizer
    
    def _custom_states_class(self):
        return CustomTrainerStates
        
    def _build_dataloader(self, mode='train'):
        dataset = self._dataset_class()(self.opt, self.accelerator, mode=mode)
        dataloader = get_dataloader(dataset, self.opt)
        return dataloader
        
    def _build_metrics(self, mode='train'):
        metrics = Seq2SeqMetrics(self.opt, mode=mode, accelerator=self.accelerator)
        return metrics
    
    def _optimizer_class(self):
        deepspeed_states = AcceleratorState().deepspeed_plugin
        if deepspeed_states.deepspeed_config['zero_optimization']['offload_optimizer']['device'] in ('none', None):
            return optim.AdamW
        return DeepSpeedCPUAdam
        
    def _build_optimizer(self):
        lr = self.opt.lr
        if self.opt.scheduler == 'noam':
            lr = calculate_noam_lr(dmodel=self.model.config.hidden_size, step=self.opt.warmup_steps)
            write_log_info_on_rank0(f'Noam scheduler override the learning rate to {lr:.3e}')
        
        params = get_optimizer_grouped_parameters(self.model, weight_decay=self.opt.weight_decay)
        optimizer = self._optimizer_class()(params, lr=lr, eps=self.opt.eps, betas=(self.opt.beta1, self.opt.beta2))
            
        return optimizer
        
    def _build_scheduler(self):
        if self.opt.scheduler == 'invsqrt':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                    lr_lambda=invsqrt_scheduler(self.opt.warmup_steps))
        elif self.opt.scheduler == 'noam':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                    lr_lambda=noam_scheduler(dmodel=self.model.config.hidden_size, 
                                                                             warmup_steps=self.opt.warmup_steps))
        elif self.opt.scheduler == 'warmup':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                    lr_lambda=warmup_scheduler(warmup_steps=self.opt.warmup_steps,
                                                                               min_factor=0.1))
        elif self.opt.scheduler == 'constant':
            lr = self.opt.lr
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                    lr_lambda=lambda step: 1.)
        elif self.opt.scheduler == 'reduceonplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, 
                                                             factor=self.opt.reduce_factor, 
                                                             patience=self.opt.reduce_patience,
                                                             mode='min' if self.opt.validation_metric in ('loss', 'ppl') else 'max',
                                                             min_lr=1e-6)
            self.schedule_on_valid = True
        else:
            raise ValueError
        return scheduler
    
    def _build_loss_fn(self):
        return nn.CrossEntropyLoss(ignore_index=self.tokenizer.null_token_id, reduction='none', label_smoothing=self.opt.label_smoothing)
        
    def _get_metric_obj(self, training=True):
        if training:
            return self.train_metrics
        return self.valid_metrics
    
    @torch.no_grad()
    def _record_batch_info(self, batch, mode='train'):
        batchsize = batch.get('n_exps', batch['text_vec'].size(0))
        if mode == 'train':
            self.train_metrics.record_metric_many('clen', batch['text_len'])
            self.train_metrics.record_metric_many('ctrunc', batch['text_trunc'])
            self.train_metrics.record_metric('tpb', batch['n_tokens'])
            self.train_metrics.record_metric('expb', batchsize)
            self.train_metrics.record_metric('total_exs', batchsize)
            self.custom_states.total_exps += batchsize
        elif mode == 'valid':
            self.valid_metrics.record_metric_many('clen', batch['text_len'])
            self.valid_metrics.record_metric_many('ctrunc', batch['text_trunc'])
            self.valid_metrics.record_metric('total_exs', batchsize)
        else:
            raise ValueError
    
    # calcucate loss and other metrics with model outputs, return loss
    def _criterion(self, model_output: Tuple[torch.Tensor, ...], batch: Dict[str, Any], return_output=False, training=True):
        null_idx = self.tokenizer.null_token_id
        
        scores, preds, *_ = model_output
        labels: torch.LongTensor = batch['label_vec']
        
        # calculate loss
        score_view = scores.reshape(-1, scores.size(-1)) # bs * num_tokens, vocab_size
        loss = self.loss_fn(score_view if not self.fp32_loss else score_view.to(torch.float32), labels.reshape(-1)).sum()
        if self.fp32_loss:
            loss = loss.to(scores.dtype) # cast back
        
        # calculate token acc
        notnull = labels.ne(null_idx)
        target_tokens = notnull.sum()
        correct = ((labels == preds) * notnull).sum()
        
        # average losses
        loss = loss / target_tokens

        # logs
        with torch.no_grad():
            obj_metrics = self._get_metric_obj(training)
            obj_metrics.record_metric('loss', loss.item())
            obj_metrics.record_metric('ppl', loss.item())
            obj_metrics.record_metric('token_acc', (correct / target_tokens).item())

        if return_output:
            return (loss, model_output)
        return loss
    
    def _run_forward(self, batch: Dict[str, Any], **kwargs):
        # print('before', batch['text_vec'].dtype)
        return self.model(decoder_input=batch['text_vec'], **kwargs)
    
    def _run_fake_forward(self):
        # a dummy forward for stage3 compatibility to avoid deadlock
        self.model(decoder_input=torch.tensor([[0]], dtype=torch.long, device=self.accelerator.device))      
        
    def _train_step(self, batch: Dict[str, Any], **kwargs):
        self.optimizer.zero_grad()
        # run forward
        assert self.model.training
        model_output = self._run_forward(batch, **kwargs)    
        # calculate loss
        loss = self._criterion(model_output, batch)
        self.accelerator.backward(loss)
        # debug
        if torch.isnan(loss) or torch.isinf(loss) or loss.abs().gt(10000.):
            logging.warn(f'strange loss {loss.item()} detected')
            # with open(f'strange_loss_batch_{self.custom_states.total_steps}_{self.accelerator.process_index}.json', 'w') as f:
            #     json.dump({'text': batch['text'], 'encoded': batch['text_encoded']}, f, ensure_ascii=False, indent=4)
        
        # clip gradients(do automatically by deepspeed)
        # if self.accelerator.sync_gradients:
        #     gnorm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        # self.train_metrics.record_metric('gnorm', float(gnorm))
        
        # update parameters
        self.optimizer.step()
        if not getattr(self, 'schedule_on_valid', False) and not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()
        
                    
    def generate_sentence(self, context: Union[str, List[str]], **kwargs):
        return self.model.generate_sentence(context, **kwargs)
    
    def _on_stop_train(self):
        return self.custom_states.total_steps >= self.max_steps or self.custom_states.patient >= self.opt.patience
    
    def _pre_epoch(self):
        # do something before each epoch
        pass
    
    def _post_epoch(self):
        # do something after each epoch
        pass
           
    def train(self):
        synchronize_if_distributed()
        write_log_info_on_rank0('Start training')
        self.model.train()
        
        while not self._on_stop_train():
            self._pre_epoch()
            for batch in self.train_loader:
                if self._on_stop_train():
                    break
                
                start_time = time.time()
                # record some info
                self._record_batch_info(batch, mode='train')
                to_cuda(batch)
                # perform a step of train
                self._train_step(batch)
                del batch
                
                # record
                cost_time = time.time() - start_time
                self.train_metrics.record_metric('ups', 1. / cost_time)
                if hasattr(self.scheduler, 'get_last_lr'):
                    lr = self.scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]['lr']
                self.train_metrics.record_metric('lr', lr)
                self.custom_states.total_steps += 1
                
                # print metrics for every 50 steps
                need_reset = False
                if self.custom_states.total_steps % self.print_interval == 0:
                    metrics = self.train_metrics.all_gather_metrics()
                    self.train_metrics.write_tensorboard(self.custom_states.total_steps, gathered_metrics=metrics)
                    self.train_metrics.display(self.custom_states.total_steps, self.train_size, gathered_metrics=metrics)
                    need_reset = True
                    
                # do evaluation for every save_freq steps
                if self.custom_states.total_steps % self.save_freq == 0:
                    eval_score, _ = self.evaluate()
                    self.model.train()
                    
                    if any(kwd in self.validation_metric for kwd in ('loss', 'ppl')):
                        # if smaller is better
                        eval_score = -eval_score
                        
                    # save checkpoint
                    is_best = eval_score > self.custom_states.best_score
                    if is_best:
                        self.custom_states.patient = 0
                        self.custom_states.best_score = eval_score
                        write_log_info_on_rank0(f'Achieved the best score {abs(eval_score)}')
                    else:
                        self.custom_states.patient += 1
                        write_log_info_on_rank0(f'Did not beat the best score {abs(self.custom_states.best_score)}, patient {self.custom_states.patient}')
                        
                    self._save_checkpoint(is_best=is_best, total_steps=self.custom_states.total_steps)
        
                if need_reset:
                    self.train_metrics.reset(no_reset=self.no_reset_metric_names)
            self._post_epoch()
                    
    def get_generate_batch(self, batch: Dict[str, Any]):
        if 'text_vec_no_label' in batch:
            return {'text_vec': batch['text_vec_no_label']}
        return batch
    
    @torch.no_grad()   
    def evaluate(self, datatype='valid', **kwargs) -> Tuple[float, List]:
        assert datatype in ('valid', 'test')
        
        start_time = time.time()
        valid_dataloader = self._build_dataloader(mode=datatype)
        valid_size = len(valid_dataloader.dataset)
        generated_sents: List[Tuple[float, str, str, str]] = []
        write_log_info_on_rank0(f'Start evaluation on {datatype} data, generation mode: {not self.skip_generation_on_eval}')
        self.model.eval()
        
        for step, batch in enumerate(valid_dataloader):
            # record some info
            self._record_batch_info(batch, mode='valid')
            to_cuda(batch)
            # run a forward                    
            model_output = self._run_forward(batch, **kwargs)
            # calculate and record non-generation metrics like loss and ppl
            self._criterion(model_output, batch, training=False)
            if is_deepspeed_zero3_enabled():
                synchronize_forward_on_stage3(False, self._run_fake_forward)
            
            # generate sequences if required
            if not self.skip_generation_on_eval and len(generated_sents) < self.opt.num_examples:
                generation_output, all_candidates = self.model.generate(self.get_generate_batch(batch))
                
                assert len(generation_output) == len(batch['label']) == len(batch['text'])
                for (log_prob, seq), gold_sent, context in zip(generation_output, batch['label'], batch['text']):
                    # convert to string
                    seq_str = self.tokenizer.vec2txt(seq, skip_special=True)
                    # print(f'{seq_str}\n{gold_sent}\n-------------')
                    generated_sents.append((log_prob, context, seq_str, gold_sent))
                    
                    # record metrics
                    self.valid_metrics.record_metric('f1', (seq_str, [gold_sent]))
                    for i in (1, 2, 3, 4):
                        self.valid_metrics.record_metric(f'bleu-{i}', (seq_str, [gold_sent]))
                    for i in (1, 2, 'l'):
                        self.valid_metrics.record_metric(f'rouge-{i}', (seq_str, [gold_sent]))
                    for i in (1, 2):
                        self.valid_metrics.record_metric(f'dist-{i}', seq_str)
                    self.valid_metrics.record_metric(f'cider', (seq_str, [gold_sent]))
            
            # if (step + 1) % 100 == 0 and not self.skip_generation_on_eval:
            #     self.valid_metrics.display(step, data_size=valid_size)
            #     write_log_info_on_rank0('Generation samples: ')
            #     for sample in generated_sents[-5:]:
            #         print(sample)
        
        if is_deepspeed_zero3_enabled():
            synchronize_forward_on_stage3(True, self._run_fake_forward)
        # log info
        metrics = self.valid_metrics.all_gather_metrics()
        self.valid_metrics.display(self.custom_states.total_steps, gathered_metrics=metrics)
        self.valid_metrics.write_tensorboard(self.custom_states.total_steps, gathered_metrics=metrics)
        self.valid_metrics.flush()
        validation_score = metrics[self.validation_metric]
        if getattr(self, 'schedule_on_valid', False) and getattr(self, 'scheduler', None) is not None:
            self.scheduler.step(validation_score)
        self.valid_metrics.reset(no_reset=[])
        
        write_log_info_on_rank0(f'Evaluation completed in {(time.time() - start_time):.2f} seconds')
        return validation_score, None
    
    def _get_extra_inference_info(self, batch: Dict[str, Any]) -> Tuple[List[Any], ...]:
        # Any more information to be recorded in inference
        return ()
    
    def _parse_extra_inference_info(self, others: List[Any]) -> Tuple[Any, ...]:
        # elements in others match each elements return by _get_extra_inference_info
        return ()
    
    def inference(self, datatype='test', num_examples: int=999999) -> List:
        # TODO: multi-gpu inference
        write_log_info_on_rank0(f'Start inference')
        start_time = time.time()
        dataloader = self._build_dataloader(mode=datatype)
        generated_sents: List[Tuple[float, str, str, str]] = []
        
        self.model.requires_grad_(False)
        self.model.eval()
        for step, batch in enumerate(dataloader):
            to_cuda(batch)
            generation_output, all_candidates = self.model.generate(self.get_generate_batch(batch))
            
            assert len(generation_output) == len(batch['text'])
            # print(batch['text_vec'])
            for (log_prob, seq), context, candidates, *others in zip(*((generation_output, batch['text'], all_candidates) + self._get_extra_inference_info(batch))):
                # convert to string
                # print(context)
                seq_str = self.tokenizer.vec2txt(seq, skip_special=True).strip()
                # handle sentence that exceed max_ts and not completely generated
                if seq[-1] not in self.model.terminate_idxs:
                    seq_str = cut_uncomplete_sentence(seq_str)
                    
                candidates_str = '\n'.join(self.tokenizer.vec2txt(candid, skip_special=True).strip() if candid[-1] in self.model.terminate_idxs else \
                                            cut_uncomplete_sentence(self.tokenizer.vec2txt(candid, skip_special=True).strip()) for _, candid in candidates)
                other_info = self._parse_extra_inference_info(others)
                generated_sents.append((log_prob, context, seq_str, candidates_str) + other_info)
                
            # check the num of samples already generated
            if len(generated_sents) >= num_examples:
                break
            
        write_log_info_on_rank0(f'Inference completed in {(time.time() - start_time):.2f} seconds')
        return generated_sents
    