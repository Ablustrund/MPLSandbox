import argparse

def parse_args(*args):
    parser = argparse.ArgumentParser(description='generation model config.')
    
    # Model (chitchat) args
    parser.add_argument('--hf_model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='Hugging model name used to load vocabs, configs and pretained models')
    parser.add_argument('--init_from_hf_pretrain', action='store_true', help='whether to load weights from hugging face')
    parser.add_argument('--delimiter', type=str, default='\n', help='delimiter to seperate dialog history')
    parser.add_argument('--model_type', type=str, default='llama', help='model type')
    # parser.add_argument('--vocab_path', type=str, default=None, help='a customized vocabulary to override the default huggingface vocab')
    # parser.add_argument('--hidden_size', type=int, default=None, help='customize model if "init_from_hf_pretrain" is False')
    # parser.add_argument('--num_heads', type=int, default=None, help='customize model if "init_from_hf_pretrain" is False')
    # parser.add_argument('--num_layers', type=int, default=None, help='customize model if "init_from_hf_pretrain" is False')
    # parser.add_argument('--intermediate_size', type=int, default=None, help='customize model if "init_from_hf_pretrain" is False')
    # parser.add_argument('--layernorm_type', type=str, default='post')
    # parser.add_argument('--n_layers_freeze', type=int, default=0)
    
    # GPT (decode-only model) args
    # parser.add_argument('--separate_context_response', action='store_true', help='if true, calculate the loss of last utterance (response) only')
    parser.add_argument('--separate_prompt', type=str, default='P1: |P2: ')
    parser.add_argument('--no_prompt', action='store_true', help='Enable pure next token prediction pretraining')
    # parser.add_argument('--force_p2_response', action='store_true', help='When separate_prompt, set the prefix of response is always p2 (the even number of uttrs)')
    
    # Different task args
    parser.add_argument('--add_kd', action='store_true', help='add knowledge part for each dialog')
    parser.add_argument('--kd_len', type=int, default=256, help='max length of knowledge')
    parser.add_argument('--add_role', action='store_true', help='train role chat task')
    parser.add_argument('--multi_role', action='store_true')
    
    # Checkpoint args
    parser.add_argument('--model_file', type=str, default='./ckpts', help='checkpoint path, used for save model and continuous training from a breakpoint')
    parser.add_argument('--init_model', type=str, default=None, help='checkpoint used to initialize the model, used for fine-tuning')
    parser.add_argument('--init_model1', type=str, default=None, help='checkpoint used to initialize the model, used for fine-tuning')
    parser.add_argument('--init_model2', type=str, default=None, help='checkpoint used to initialize the model, used for fine-tuning')
    parser.add_argument('--hdfs_ckpt_path', type=str, default=None, help='upload/download checkpoints to/from HDFS')
    
    # Dataset args
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--context_truncate', type=int, default=2048, help='max length for history')
    parser.add_argument('--label_truncate', type=int, default=None, help='max length for response')
    parser.add_argument('--dynamic_batching', action='store_true', help='perform dynamic batching instead of fixed batchsize to accelerate training. the max tokens for each batch equals to batchsize * (context_trunc + label_trunc)')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset folder path')
    parser.add_argument('--use_chunk_data', action='store_true', help='data that cannot fit in the memory and split into chunks')
    parser.add_argument('--num_workers', type=int, default=1, help='>0 for multiprocessing data loader')
    parser.add_argument('--num_prefetch', type=int, default=32, help='num of batches for each prefetch process')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--openai_style_prompt', action='store_true', help='use openai style instead of xP3 style prompt')
    parser.add_argument('--belle_style_prompt', action='store_true', help='use belle style prompt')
    parser.add_argument('--chatglm_style_prompt', action='store_true', help='use chatglm style prompt')
    parser.add_argument('--plug_style_prompt', action='store_true', help='use chatplug style prompt')
    parser.add_argument('--merge_role_prompts', action='store_true', help='ROLE: merge inline and init role information')
    parser.add_argument('--no_split_dialog', action='store_true')
    
    # Inference args
    parser.add_argument('--beam_size', type=int, default=1, help='num of candidates for decoding')
    parser.add_argument('--beam_groups', type=int, default=1)
    parser.add_argument('--group_delay', type=int ,default=1, help='num of steps before applying grouped beam search')
    parser.add_argument('--max_ts', type=int, default=128, help='max tokens to generate.')
    parser.add_argument('--temperature', type=float, default=1., help='temperature to rescale the logits before softmax.')
    parser.add_argument('--repetition_penalty', type=float, default=1., help='avoid from generation repetition tokens')
    parser.add_argument('--context_repetition_penalty', type=float, default=1., help='avoid from generation repetition tokens')
    parser.add_argument('--beam_min_length', type=int, default=0, help='minimal length to generate')
    parser.add_argument('--inference', type=str, default='beam', help='decoding algorithm')
    parser.add_argument('--topp', type=float, default=0.9, help='p for nucleus sampling')
    parser.add_argument('--beam_length_penalty', type=float, default=1., help='rescore the generation outputs to penalize short sequences')
    parser.add_argument('--length_penalty_version', type=str, default='eva')
    parser.add_argument('--bleu_backend', type=str, default='sacre', help='backend used for calculating BLEU')
    parser.add_argument('--bleu_level', type=str, default='sentence')
    parser.add_argument('--cider_sigma', type=float, default=15., help='sigma for CIDEr')
    parser.add_argument('--lang', type=str, default='zh', help='language the model trained on')
    parser.add_argument('--num_examples', type=int, default=999999, help='num of examples to generate')
    parser.add_argument('--no_repeat_ngram', type=int, default=-1, help='ngrams that are penalized for second time generation')
    parser.add_argument('--ngram_blacklist', type=str, default=None, help='a blacklist of ngrams forbid for generation. TODO. ')
    parser.add_argument('--no_history', action='store_true', help='do not record dialog history for interactive inference')
    parser.add_argument('--use_huggingface_generate', action='store_true', help='use huggingface generate() interface instead')
    
    # Training args
    parser.add_argument('--skip_generation', action='store_true', help='limited metrics for faster evaluation')
    parser.add_argument('--train_steps', type=int, default=999999, help='max train steps')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='steps for learning rate warmup')
    # parser.add_argument('--grad_norm', type=float, default=1., help='max norm of gradients')
    parser.add_argument('--save_freq', type=int, default=1000, help='save checkpoint for every num of steps')
    parser.add_argument('--validation_metric', type=str, default='loss', help='metric to select the best model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. will be ignored if using noam scheduler')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--beta2', type=float, default=0.995, help='adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='optimizer eps')
    parser.add_argument('--weight_decay', type=float, default=0., help='l2 weight decay')
    parser.add_argument('--scheduler', type=str, default='invsqrt', help='learning rate scheduler')
    parser.add_argument('--reduce_factor', type=float, default=0.5, help='reduceonplateau args')
    parser.add_argument('--reduce_patience', type=int, default=0, help='reduceonplateau args')
    parser.add_argument('--patience', type=int, default=999999, help='stop train patient')
    parser.add_argument('--tensorboard_logdir', type=str, default=None, help='path to write tensorboard logs')
    parser.add_argument('--label_smoothing', type=float, default=0., help='label smoothing rate for nll loss')
    parser.add_argument('--gradient_checkpoint', action='store_true', help='enable gradient checkpointing during training, which can expand almost 4x batchsize')
    # parser.add_argument('--8bit_optim', action='store_true')
    parser.add_argument('--stable_embedding', action='store_true')
    parser.add_argument('--fp32_loss', action='store_true', help='use fp32 to calculate cross-entropy loss, enable when numeric stability problem occurs')
    
    # Self-chat args
    # parser.add_argument('--selfchat_turns', type=int, default=9, help='num of turns for self-chat')
    # parser.add_argument('--selfchat_datasource', type=str, default=None, help='source of data for self-chat. None means stdin')
    # parser.add_argument('--selfchat_return_topk', action='store_true', help='get all candidates instead of top1')
    
    # tsp args
    parser.add_argument('--tsp_build_prob', type=float, default=0., help='the prob of each sample that will be built as a tsp positive sample online')
    
    # RM args
    parser.add_argument('--sampling_offtopic_prob', type=float, default=0., help='prob to sample offtopic response as negative sample')
    parser.add_argument('--reward_lm_loss_factor', type=float, default=0., help='calculate lm loss on rm model')
    
    # RLHF args
    parser.add_argument('--n_rollouts', type=int, default=128, help='num of responses to sample per iter')
    parser.add_argument('--n_candidates', type=int, default=1)
    parser.add_argument('--rollout_batch_size', type=int, default=4)
    parser.add_argument('--clip_reward', type=float, default=10.)
    parser.add_argument('--ref_mean', type=float, default=None)
    parser.add_argument('--ref_std', type=float, default=None)
    parser.add_argument('--pg_clip', type=float, default=0.2)
    parser.add_argument('--value_clip', type=float, default=0.2)
    parser.add_argument('--vf_loss_weight', type=float, default=1.)
    parser.add_argument('--init_actor', type=str, default=None)
    # parser.add_argument('--init_critic', type=str, default=None)
    parser.add_argument('--init_reward', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--beta', type=float, default=0.02)
    # parser.add_argument('--ema', type=float, default=0.992)
    parser.add_argument('--rlhf_logdir', type=str, default='tmp')
    
    parser.add_argument('--debug', action='store_true', help='debug')
    # additional args not handled here
    for func in args:
        if callable(func):
            func(parser)
    
    args = parser.parse_args()
    return args
    
    