from collections import OrderedDict
import torch
import torch.nn as nn
import time, os
from typing import Dict, Any, Tuple, List

from .ppo_datahelper import *
from data_helper import get_dataloader
from .ppo_utils import RunningMoments, logprobs_from_logits, whiten
from trainer import Seq2SeqTrainer, CustomTrainerStates
from utils import *
from metric import MeanMetric, RealtimeMetric, SumMetric
from accelerate import Accelerator
import deepspeed
from transformers.deepspeed import is_deepspeed_zero3_enabled


class RLHFTrainableModelWrapper(nn.Module):
    def __init__(self, policy_model: nn.Module, critic_model: nn.Module, reward_model: nn.Module = None, ref_model: nn.Module = None) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.reward_model = reward_model
        self.ref_model = ref_model

        if reward_model != None:
            self.reward_model.eval()
            self.reward_model.requires_grad_(False)

        if ref_model != None:
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
    
    def forward(self, inputs, **kwargs):
        return self.policy_model(decoder_input=inputs, **kwargs), self.critic_model(decoder_input=inputs, rank_all=True, **kwargs)
    
    def train(self, mode=True):
        self.policy_model.train(mode)
        self.critic_model.train(mode)
        
    def eval(self):
        self.policy_model.eval()
        self.critic_model.eval()
        if self.reward_model != None:
            self.reward_model.eval()
        if self.ref_model != None:
            self.ref_model.eval()


class PPOTrainer(Seq2SeqTrainer):
    def __init__(self, opt, policy_model: nn.Module, ref_model: nn.Module, critic_model: nn.Module, reward_model: nn.Module, accelerator: Accelerator, **kwargs) -> None:
        self.opt = opt
        self.no_reset_metric_names = ['total_exs','step_pass_rate'] # metrics won't be reset for every 50 steps
        self.print_interval = opt.n_rollouts // opt.batch_size
        self.eval_only = kwargs.get('eval_only', False)

        self.num_rollouts: int = opt.n_rollouts
        self.num_rollout_candidates: int = opt.n_candidates
        self.clip_reward: float = opt.clip_reward
        self.running = RunningMoments(accelerator)
        self.ref_mean: float = opt.ref_mean
        self.ref_std: float = opt.ref_std
        self.clip_pg: float = opt.pg_clip
        self.clip_value: float = opt.value_clip
        self.vf_coef: float = opt.vf_loss_weight
        self.beta: float = opt.beta
        
        self.model = RLHFTrainableModelWrapper(policy_model=policy_model, critic_model=critic_model)
        self.accelerator = accelerator
        
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.train_metrics = self._build_metrics('train')
        self.valid_metrics = self._build_metrics('eval_skip_generation')
        self.tokenizer = self._tokenizer_class()(opt)
        
        self.custom_states = self._custom_states_class()()
        self.max_steps: int = opt.train_steps
        self.save_freq = opt.save_freq
        self.skip_generation_on_eval: bool = opt.skip_generation
        self.save_path = opt.model_file
        self.validation_metric = opt.validation_metric
        
        self.replay_buffer = [] #这个是关键，是算一堆GAE中的东西并存进去
        self.train_loader = None

        self.prompt_dataset = self._prompt_dataset_class()(self.opt, self.accelerator, mode='train')
        self.prompt_loader = get_dataloader(self.prompt_dataset, self.opt)
        self.train_size = len(self.prompt_loader.dataset)
        self.prompt_loader = iter(self.prompt_loader)

        print(f"DEBUG: prompt_loader {self.prompt_loader}\n")

        
        self.accelerator.register_for_checkpointing(self.custom_states)

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        
        # load checkpoint
        self.load_from_checkpoint = False
        if self.opt.model_file and os.path.exists(self.opt.model_file):
            self.load_from_checkpoint = True
            self._load_checkpoint(self.opt.model_file, strict=True)
        
        # get unwrapped trainable model
        self.policy_model = self.accelerator.unwrap_model(self.model).policy_model
        self.critic_model = self.accelerator.unwrap_model(self.model).critic_model

        # get untrainable model
        eval_ds_config = get_eval_ds_config(offload=True)
        self.reward_model = reward_model
        self.ref_model, *_ = deepspeed.initialize(model=ref_model, config=eval_ds_config)
        self.ref_model.eval()
        
        self.debug = opt.debug
            
        self.post_init()
        synchronize_if_distributed()
        
    def _build_metrics(self, mode='train'):
        metrics = super()._build_metrics(mode)
        metrics.remove_useless_metric('ppl')
        metrics.remove_useless_metric('token_acc')

        metrics.add_additional_metric('rewards', MeanMetric())
        metrics.add_additional_metric('r1ward_num', MeanMetric())

        metrics.add_additional_metric('introductory', SumMetric())
        metrics.add_additional_metric('interview', SumMetric())
        metrics.add_additional_metric('competition', SumMetric())

        if mode == 'train':
            # if self.ref_mean is None or self.ref_std is None:
            #     metrics.add_additional_metric('reward_mean', MeanMetric())
            #     metrics.add_additional_metric('reward_std', MeanMetric())
            metrics.add_additional_metric('step_pass_rate',MeanMetric())
            metrics.add_additional_metric('approx_kl', MeanMetric())
            metrics.add_additional_metric('ref_kl', MeanMetric())
            metrics.add_additional_metric('returns', MeanMetric())
            metrics.add_additional_metric('advantages', MeanMetric())
            metrics.add_additional_metric('ratio', MeanMetric())
            metrics.add_additional_metric('pg_clip', MeanMetric())
            metrics.add_additional_metric('vf_clip', MeanMetric())
            metrics.add_additional_metric('pg_loss', MeanMetric())
            metrics.add_additional_metric('vf_loss', MeanMetric())
        metrics.add_additional_metric('llen', MeanMetric())
        return metrics

    def _group_optim_params(self, no_decay_name_list=["bias", "LayerNorm.weight"]):
        optimizer_grouped_parameters = []
        def _group_parms(model, submodel_name, weight_decay, lr, eps):
            params = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if (not any(nd in n
                                    for nd in no_decay_name_list) and p.requires_grad and submodel_name in n)
                    ],
                    "weight_decay": weight_decay,
                    "lr": lr,
                    "eps": eps,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if (any(nd in n
                                for nd in no_decay_name_list) and p.requires_grad and submodel_name in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                    "eps": eps,
                },
            ]
            return params
        optimizer_grouped_parameters.extend(_group_parms(self.model, 'policy_model.', self.opt.weight_decay, self.opt.lr, self.opt.eps))
        optimizer_grouped_parameters.extend(_group_parms(self.model, 'critic_model.', self.opt.weight_decay, self.opt.lr * 3, 1e-8))
        return optimizer_grouped_parameters

    def _build_optimizer(self):
        params = self._group_optim_params()
        optimizer = self._optimizer_class()(params, eps=self.opt.eps, betas=(self.opt.beta1, self.opt.beta2))
        return optimizer

    def _build_dataloader(self, mode):
        raise NotImplementedError
    
    def _prompt_dataset_class(self):
        return DialogPromptDataset
    
    def _replay_dataset_class(self):
        return DialogReplayDataset
    
    def _strip_pad(self, seq: List[int]):
        return [tok for tok in seq if tok != self.tokenizer.null_token_id]
    
    def _run_reward_forward(self, resp_vec_sampled, resps, batch, bsz, **kwargs):
        return self.reward_model.forward(resp_vec_sampled, resps, batch, bsz, **kwargs)
    
    def _run_policy_forward(self, inputs, **kwargs):
        return self.policy_model(decoder_input=inputs, **kwargs)
    
    # def _run_policy_fake_forward(self):
    #     return self.policy_model(decoder_input=torch.tensor([[0]], dtype=torch.long, device=self.accelerator.device))
    
    def _run_ref_forward(self, inputs, **kwargs):
        return self.ref_model(decoder_input=inputs, **kwargs)
    
    def _run_critic_forward(self, inputs, **kwargs):
        return self.critic_model(decoder_input=inputs, rank_all=True, **kwargs)
    
    def _run_forward(self, batch: Dict[str, Any], **kwargs):
        return self.model(batch['text_vec'], **kwargs)
    
    def _extract_context_candidates(self, context: List[List[int]], candidates: List[List[Tuple[float, List[int]]]]):
        assert len(context) == len(candidates), f'Batchsize not match {len(context)} & {len(candidates)}'
        all_context, all_resp = [], []
        for ctx, candidate in zip(context, candidates):
            ctx = self._strip_pad(ctx)
            for _, resp in candidate:
                resp = self._strip_pad(resp)
                if resp[-1] != self.tokenizer.end_token_id:
                    logging.warn(f'Found too long generated response')
                all_context.append(ctx.copy())
                all_resp.append(resp)

        all_context_resp = [c + r for c, r in zip(all_context, all_resp)]
        return all_context, all_resp, all_context_resp

    def _record_batch_info(self, batch, mode):
        super()._record_batch_info(batch, mode=mode)
        if mode == 'train':
            self.train_metrics.record_metric_many('llen', batch['label_len'])
    
    def _save_checkpoint(self, is_best: bool, total_steps: int, **kwargs):
        best_model_path = os.path.join(self.save_path, 'best_model')
        steps_model_path = os.path.join(self.save_path, '{}_steps'.format(total_steps))

        unwrapped_model = self.policy_model
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        if is_deepspeed_zero3_enabled():
            state_dict = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                filtered = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('policy_model.'):
                        filtered[k[13:]] = v
                state_dict = filtered
        else:
            state_dict = self.accelerator.get_state_dict(unwrapped_model)

        if is_best:
            unwrapped_model.save_pretrained(
                best_model_path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )
            logging.info(f'Saved best model to {best_model_path}')

        unwrapped_model.save_pretrained(
            steps_model_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
        )
        logging.info(f'Saved model of {total_steps} steps to {steps_model_path}')
        
        synchronize_if_distributed()
        # save all training states
        self.accelerator.save_state(self.save_path)
        # upload checkpoints to hdfs
        # self._upload_checkpoint(is_best=is_best)
        
    def _format_experience_log(self, log):
        log = copy.deepcopy(log)
        for sample in log:
            for k, v in sample.items():
                if isinstance(v, list):
                    sample[k] = str(v)
        return log
        
    @torch.no_grad()
    def make_experiences(self):
        logging.info(f"Start to sample experiences with num_rollouts = {self.num_rollouts} / GPU where {self.num_rollout_candidates} each prompt")
        start_time = time.time()
        self.model.eval()
        synchronize_if_distributed()
        
        while len(self.replay_buffer) < self.num_rollouts:
            # get a batch
            batch: Dict[str, Any] = next(self.prompt_loader)
            to_cuda(batch)
            context_vec = batch['text_vec'].tolist()
            
            # sample responses
            outputs, candids = self.policy_model.generate(batch, beam_size=self.num_rollout_candidates) #candids是候选的输出数量
            assert len(context_vec) == len(candids)
            
            context_vec_sampled, resp_vec_sampled, sampled_vec = self._extract_context_candidates(context_vec, candids)
            sampled_vec = torch.tensor(pad_sequences(sampled_vec, pad_value=self.tokenizer.null_token_id, pad_left=True), 
                                                     dtype=torch.long, device=self.accelerator.device)
            bsz = sampled_vec.size(0)
            
            # calculate & normalize reward
            resps = [self.tokenizer.vec2txt(resp_vec_sampled[i]) for i in range(bsz)]

            rewards, pass_list, introductory, interview, competition = self._run_reward_forward(resp_vec_sampled, resps, batch, bsz)
            rewards = rewards.cpu()
            self.train_metrics.record_metric_many('rewards', rewards.tolist())
            self.train_metrics.record_metric_many('r1ward_num', pass_list)
            self.train_metrics.record_metric_many('introductory', introductory)
            self.train_metrics.record_metric_many('interview', interview)
            self.train_metrics.record_metric_many('competition', competition)
            self.train_metrics.record_metric_many('step_pass_rate', pass_list)

            if self.ref_mean is None or self.ref_std is None:
                # rewards = whiten(rewards, mask=None, accelerator=self.accelerator).cpu()
                rewards_mean, rewards_std = self.running.update(rewards)
                rewards /= self.running.std # do not -= mean since advantage will be normalized again
                logging.info(f"Running mean: {self.running.mean}, std: {self.running.std}")
                # self.train_metrics.record_metric('reward_mean', rewards_mean)
                # self.train_metrics.record_metric('reward_std', rewards_std)
            else:
                rewards /= self.ref_std
                
            if self.clip_reward > 0.:
                rewards = torch.clip(rewards, -self.clip_reward, self.clip_reward)
                
                
            # Precompute logprobs, values
            ref_logits, *_ = self._run_ref_forward(sampled_vec)
            logits, *_ = self._run_policy_forward(sampled_vec)
            values, *_ = self._run_critic_forward(sampled_vec)
            torch.cuda.empty_cache()
            assert ref_logits.size(1) == logits.size(1) == values.size(1), f'{ref_logits.size()}, {logits.size()}, {values.size()}'
            
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], sampled_vec[:, 1:])
            logprobs = logprobs_from_logits(logits[:, :-1, :], sampled_vec[:, 1:])
            values = values[:, :-1]
            
            kl_penalty = (-self.beta * (logprobs - ref_logprobs)).cpu()
            
            # gather samples
            for i in range(bsz):
                resp_length = len(resp_vec_sampled[i])
                penalized_rewards = kl_penalty[i].clone()
                penalized_rewards[-1] += rewards[i] # 给定任何文本，奖励模型将为最后一个token分配一个标量奖励值
                self.train_metrics.record_metric('ref_kl', (logprobs[i][-resp_length:] - ref_logprobs[i][-resp_length:]).mean().item())
                
                sample = {
                    'id': batch['ids'][i],
                    'start_state': batch['start_state'][i],
                    'context_vec': context_vec_sampled[i],
                    'context': self.tokenizer.vec2txt(context_vec_sampled[i]),
                    'resp_vec': resp_vec_sampled[i],
                    'resp': resps[i],
                    'reward': penalized_rewards[-resp_length:].tolist(),
                    'values': values[i][-resp_length:].tolist(),
                    'ref_logprobs': ref_logprobs[i][-resp_length:].tolist(),
                    'logprobs': logprobs[i][-resp_length:].tolist(),
                }
                self.replay_buffer.append(sample)

        
        if self.accelerator.is_main_process:
            log_dir = f"tmp/{self.opt.model_file.split('/')[-1]}/experiences"
            os.makedirs(log_dir, exist_ok=True)
            json.dump(self._format_experience_log(self.replay_buffer), open(f'{log_dir}/experiences_{self.custom_states.total_steps}.json', 'w'), ensure_ascii=False, indent=4)

        # if self.accelerator.is_main_process:
        #     os.makedirs('tmp', exist_ok=True)
        #     json.dump(self._format_experience_log(self.replay_buffer), open(f'tmp/experiences_{self.custom_states.total_steps}.json', 'w'), ensure_ascii=False, indent=4)  
        logging.info(f'Sampled {len(self.replay_buffer)} samples in {(time.time() - start_time):.2f} seconds')
        self.model.train()
        print("----------------------------------------------------------------------------generate break-----------------------------------------------------------------------------------------")
        # breakpoint()  
        
    def _criterion(self, model_output: Tuple[torch.Tensor, ...], batch: Dict[str, Any], return_output=False, training=True):
        policy_output, critic_output = model_output
        policy_logits, *_ = policy_output
        values, *_ = critic_output
        values = values[:, :-1]
        
        loss_mask = batch['loss_mask']
        loss_mask = loss_mask[:, 1:]
        old_values = batch['values']
        old_logprobs = batch['logprobs']
        advantages = batch['advantages']
        returns = batch['returns']
        # advantages = whiten(advantages, loss_mask, accelerator=self.accelerator)
        n = loss_mask.sum()
        
        logprobs = logprobs_from_logits(policy_logits[:, :-1, :], batch['text_vec'][:, 1:]) * loss_mask
        
        # vf loss
        values_clipped = torch.clamp(
            values,
            old_values - self.clip_value,
            old_values + self.clip_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * loss_mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * loss_mask) / n
        
        # pg loss
        # if self.accelerator.is_main_process:
        #     print(logprobs[0], '\n', old_logprobs[0])
        log_ratio = (logprobs - old_logprobs) * loss_mask
        ratio = torch.exp(log_ratio)
        with torch.no_grad():
            approx_kl = torch.sum((ratio - 1) - log_ratio) / n
            
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.clip_pg,
            1.0 + self.clip_pg,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * loss_mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * loss_mask) / n
        
        loss = pg_loss + self.vf_coef * vf_loss
        
        with torch.no_grad():
            obj_metrics = self._get_metric_obj(training)
            obj_metrics.record_metric('loss', loss.item())
            obj_metrics.record_metric('pg_loss', pg_loss.item())
            obj_metrics.record_metric('vf_loss', vf_loss.item())
            obj_metrics.record_metric('pg_clip', pg_clipfrac.item())
            obj_metrics.record_metric('vf_clip', vf_clipfrac.item())
            obj_metrics.record_metric('approx_kl', approx_kl.item())
            obj_metrics.record_metric('advantages', (advantages.mul(loss_mask).sum() / n).item())
            obj_metrics.record_metric('returns', (returns.mul(loss_mask).sum() / n).item())
            obj_metrics.record_metric('ratio', (ratio.mul(loss_mask).sum() / n).item())
            
        if return_output:
            return loss, model_output
        return loss
    
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
            outputs, candidates = self.policy_model.generate(batch, beam_size=1, **kwargs)
            _, resp_vec, output_vec = self._extract_context_candidates(batch['text_vec'].tolist(), candidates)
            if self.debug:
                logging.info(self.tokenizer.vec2txt(output_vec[-1]))
            
            output_vec = torch.tensor(pad_sequences(output_vec, pad_value=self.tokenizer.null_token_id, pad_left=True), 
                                                     dtype=torch.long, device=self.accelerator.device)
            
            bsz = output_vec.size(0)
            resps = [self.tokenizer.vec2txt(resp_vec[i]) for i in range(bsz)]
            rewards, pass_list, introductory, interview, competition = self._run_reward_forward(resp_vec, resps, batch, bsz, mode='valid')
            
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
            
            
    def _pre_epoch(self):
        self.make_experiences() #这里开始采样了，采样后获得replay buffer的train loader
        for a in self.prompt_dataset.archives:
            print(a.items)
            print('pre epoch')
        self.replay_dataset = self._replay_dataset_class()(self.replay_buffer, self.opt, self.accelerator, self.prompt_dataset.archives)
        self.train_loader = get_dataloader(self.replay_dataset, self.opt)
        print(f"DEBUG: train_loader {self.train_loader}\n")
        
    def _post_epoch(self):
        synchronize_if_distributed()
        self.train_loader = None
        self.replay_buffer.clear()
        torch.cuda.empty_cache()
    
    def train(self):
        # if not self.load_from_checkpoint:
        #     eval_score, _ = self.evaluate()
        #     self.custom_states.best_score = eval_score
            
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
                    # if self.update_level:
                    #     self.update_level = False
                    #     self.opt.level += 1
                    #     write_log_info_on_rank0(f"-----------------------update the level to {self.opt.level}-------------------------------")
                    #     self.opt.pass_rate = self.opt.pass_rate_l2
                    #     self.prompt_loader = get_dataloader(self._prompt_dataset_class()(self.opt, self.accelerator, mode='train'), self.opt)
                    #     self.prompt_loader = iter(self.prompt_loader)
                    #     synchronize_if_distributed()
                    #     self.no_reset_metric_names = ['total_exs']

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
                    self.no_reset_metric_names = ['total_exs', 'step_pass_rate']
            for a in self.replay_dataset.archives:
                print(a.items)
                print('post epoch')
            print("Address of obj1:", id(self.prompt_dataset.archives))
            print("Address of obj2:", id(self.replay_dataset.archives))
            self._post_epoch()
    
