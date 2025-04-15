from llama.llama_model import Llama
from llama.reward.llama_reward_model import LlamaRewardModel, LlamaCriticModel
from ppo.ppo_trainer import PPOTrainer
from tokenizer import HFPretrainedTokenizer as LlamaPretrainedTokenizer
from transformers import BloomConfig, LlamaConfig
from utils import *
import warnings
from config import parse_args
import os, time
import torch
import datetime

def additional_args(parser):
    parser.add_argument('--random_ratio', type=float, default=0.95)
    parser.add_argument('--archive_size', type=int, default=10)
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    opt = parse_args(additional_args)
    accelerator = setup_accelerator(split_batches=True)
    setup_deepspeed_plugin(opt)
    setup_logging()
    setup_seed()
    if opt.openai_style_prompt:
        opt.separate_prompt = 'Human: |AI: '
    elif opt.belle_style_prompt:
        opt.separate_prompt = 'Human: |Assistant: '
        opt.openai_style_prompt = True
    else:
        logging.warn(f'Prompt style is set by default as "Human: |Assistant: "')
        opt.separate_prompt = 'Human: |Assistant: '
        
    setup_prompt(opt.separate_prompt)
    write_log_info_on_rank0(opt)
    
    # huggingface style init model
    tokenizer = LlamaPretrainedTokenizer(opt)
    if opt.init_actor is None:
        opt.init_actor = opt.init_model

    logging.info(f"Load policy model from {opt.init_actor}")
    policy_model = Llama.from_pretrained(opt.init_actor, opt, tokenizer)

    # policy_model._set_gradient_checkpointing(policy_model.model, opt.gradient_checkpoint)
    # write_log_info_on_rank0(f"policy model is:\n{policy_model}", log_once=True)
    write_log_info_on_rank0(f"policy model finish!", log_once=True)

    logging.info(f"Load ref model from {opt.init_actor}")
    ref_model = Llama.from_pretrained(opt.init_actor, opt, tokenizer)
    write_log_info_on_rank0(f"ref model finish", log_once=True)

    logging.info(f"Load value model from {opt.init_reward}")
    critic_model = LlamaCriticModel.from_pretrained(opt.init_reward, opt, tokenizer)
    # critic_model._set_gradient_checkpointing(critic_model.model, opt.gradient_checkpoint)
    write_log_info_on_rank0(f"critic model finish", log_once=True)

    # logging.info(f"Load reward model from {opt.init_reward}")
    # reward_model = LlamaRewardModel.from_pretrained(opt.init_reward, opt, tokenizer)
    # write_log_info_on_rank0(f"reward model is:\n{reward_model}", log_once=True)
    reward_model = LlamaRewardModel(opt, tokenizer)
    synchronize_if_distributed()

    trainer = PPOTrainer(opt, policy_model, ref_model, critic_model, reward_model, accelerator)
    trainer.train()

    logging.info('Training finished, processes will be killed in 180 seconds')
    time.sleep(180)