import time
import logging, random
import torch
import numpy as np
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from typing import Callable, List, Union, Set, Dict, Any

_seen_logs: Set[str] = set()
accelerator: Accelerator = None

def setup_accelerator(**kwargs):
    global accelerator
    if accelerator is None:
        accelerator = Accelerator(**kwargs)
    return accelerator

def setup_deepspeed_plugin(opt):
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'] = opt.batch_size # this is a dummy value
    deepspeed_states.deepspeed_config['checkpoint'] = {'use_node_local_storage': True}

def setup_logging():
    if accelerator and accelerator.use_distributed:
        logging.basicConfig(format='%(asctime)s - ' + f'Rank: {accelerator.process_index}' + ' - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    return logger

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_separate_prompt(opt):
    if getattr(opt, 'belle_style_prompt', False):
        # opt.separate_prompt = 'Human: |Assistant: '
        opt.separate_prompt = ' | '
        opt.openai_style_prompt = True
    else:
        raise ValueError
    setup_prompt(opt.separate_prompt)

def setup_model(opt):
    from tokenizer import HFPretrainedTokenizer
    tokenizer_class = HFPretrainedTokenizer

    from llama.llama_model import Llama
    from llama.llama_trainer import LlamaTrainer
    from transformers import LlamaConfig
    model_class = Llama
    config_class = LlamaConfig
    trainer_class = LlamaTrainer

    return model_class, config_class, trainer_class, tokenizer_class

def synchronize_if_distributed():
    if accelerator.use_distributed:
        accelerator.wait_for_everyone()
        
def synchronize_forward_on_stage3(done: bool, fake_forward_fn: Callable, **kwargs):
    # synchronize to avoid deadlock on deepspeed stage3. do not call this if zero-3 is disabled
    # https://github.com/microsoft/DeepSpeed/issues/860
    if done:
        sync = 1.
        while sync > 1e-5:
            fake_forward_fn(**kwargs)
            sync = torch.tensor(0., device=accelerator.device)
            sync = accelerator.reduce(sync).item()
    else:
        sync = torch.tensor(1., device=accelerator.device)
        sync = accelerator.reduce(sync)

def write_log_info_on_rank0(msg: str, log_once=False, local=True):
    if accelerator and not accelerator.is_main_process:
        return
    if log_once:
        if msg in _seen_logs:
            return
        _seen_logs.add(msg)
    logging.info(msg)
    
def total_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def to_cuda(batch: Dict[str, Any]):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(accelerator.device, non_blocking=True)

def to_half(state_dict):
    for k in state_dict:
        if state_dict[k].dtype == torch.float32 or state_dict[k].dtype == torch.bfloat16:
            state_dict[k] = state_dict[k].half()
    return state_dict

def get_eval_ds_config(offload=None, stage=3):
    deepspeed_states = AcceleratorState().deepspeed_plugin

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        }
    }
    return {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'],
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }

def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

# ------- split sentence ---------
try:
    from jionlp.gadget.split_sentence import SplitSentence
    class MySplitSentence(SplitSentence):
        def _prepare(self):
            super()._prepare()
            self.puncs_fine.extend(['.', ',', '、'])
            self.puncs_coarse.extend(['!', '?', '.', ','])
except ModuleNotFoundError:
    class MySplitSentence:
        def __call__(self, *args: Any, **kwds: Any) -> Any:
            raise NotImplementedError

split_sentence = MySplitSentence()

def cut_uncomplete_sentence(s: str, coarse=False):
    s_coarse = ''.join(split_sentence(s, criterion='coarse')[:-1]) if coarse else ''
    s_fine = ''.join(split_sentence(s, criterion='fine')[:-1])
    return s_coarse or s_fine or s


# ------- prompt ---------
p1_prompt = None
p2_prompt = None
def setup_prompt(prompt: str):
    global p1_prompt, p2_prompt
    p1_prompt, p2_prompt = prompt.split('|')
    p1_prompt, p2_prompt = p1_prompt.strip(), p2_prompt.strip()
    write_log_info_on_rank0(f'Dialogs will be composed as \n{p1_prompt}u1\n{p2_prompt}u2\n')

def p1_prompt_():
    return p1_prompt

def p2_prompt_():
    return p2_prompt
    
# shihan dou
@DeprecationWarning
def get_separate_prompt(i: int):
    print("Deprecate@@@")
    assert p1_prompt is not None and p2_prompt is not None
    return p1_prompt if i % 2 == 0 else p2_prompt

def build_prompt(code_description: str, starter_code: str,**kwargs):
    return _build_code_prompt(code_description, starter_code, **kwargs)
    
def _build_code_prompt(code_description: str, starter_code: str, dialog_sep='\n', openai_style=None):
    # for ours and wizardcoder
    # template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{starter_code}"
    # description = '\n'.join([j for j in code_description.split("\n") if not j.startswith("#")])
    # instruction = template.format_map({
    #     'instruction': description,
    #     'starter_code': starter_code
    #     # 'function_name': code_description 
    # })

    # for deepseek-coder
    if starter_code is None:
        starter_code = ""
    prompt = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\nwrite an algorithm in python:\n{instruction}\n### Response:\n```python\n{starter_code}\n"
    begin = code_description.index("\"\"\"") + len("\"\"\"")
    end = code_description.index("\"\"\"\n")
    description = code_description[begin:end]
    instruction = prompt.format_map({
        'instruction': description,
        'starter_code': starter_code
    })
    
    
    # print(f"=====\n{instruction}\n=====")
    return instruction

# def build_code_context(scope:List[List], level, golden_solution):
#     def_pos = 1
#     for i,s in enumerate(scope):
#         if s[0] == "Function Body":
#             # print(s[1])
#             # function_name = sol[s[1]-1][:-1]
#             def_pos = i + 1
#             break
#     scope_num = len(scope) - def_pos
#     if level < 3 and scope_num > 2 :
#         if scope_num == 3:
#             level_pos = scope[1 + def_pos][1]
#         elif scope_num > 3:
#             if level == 1:
#                 level_pos = scope[scope_num//2 -1 + def_pos][1]
#             else:
#                 level_pos = scope[scope_num//4 -1 + def_pos][1]

#         golden_solution_lines = golden_solution.split('\n')
#         context_solution = '\n'.join(golden_solution_lines[scope[def_pos-1][1]:level_pos]) + '\n'
#         return context_solution

def build_code_context(scope:List[List], level, golden_solution):
    def_pos = 1
    for i,s in enumerate(scope):
        if s[0] == "Function Body":
            # print(s[1])
            # function_name = sol[s[1]-1][:-1]
            def_pos = i + 1
            break
    scope_num = len(scope) - def_pos
    if level < 3 and scope_num > 2 :
        if scope_num == 3:
            level_pos = scope[1 + def_pos][1]
        elif scope_num > 3:
            if level == 1:
                level_pos = scope[scope_num//2 -1 + def_pos][1]
            else:
                level_pos = scope[scope_num//4 -1 + def_pos][1]

        golden_solution_lines = golden_solution.split('\n')
        context_solution = '\n'.join(golden_solution_lines[:level_pos]) + '\n'
        return context_solution

def _build_chitchat_prompt(context: Union[List[str], str], openai_style=False, chatglm_style=False, dialog_sep='\n', custom_prompt=''):
    if chatglm_style:
        assert isinstance(context, list)
        
    if isinstance(context, list):
        if context[-1].startswith(p1_prompt):
            n_turns = 1
        elif context[-1].startswith(p2_prompt):
            n_turns = 2
        else:
            logging.critical(context)
            raise ValueError
        
        if chatglm_style:
            parsed_context = ''
            assert len(context) % 2 == 1, f'chatglm context should have odd length'
            for i in range(0, len(context) // 2):
                q_i, a_i = i * 2, i * 2 + 1
                assert context[q_i].startswith(p1_prompt) and context[a_i].startswith(p2_prompt)
                parsed_context += f"[Round {i}]\n{context[q_i]}\n{context[a_i]}\n"
            parsed_context += f"[Round {len(context) // 2}]\n{context[-1]}\n{p2_prompt}" 
        else:
            context = dialog_sep.join(context)
    else:
        n_turns = -1

    if custom_prompt:
        return f"{context}\n{custom_prompt}{dialog_sep}" + (get_separate_prompt(n_turns) if n_turns > 0 else '')
    elif openai_style:
        return f"{context}{dialog_sep}" + (get_separate_prompt(n_turns) if n_turns > 0 else '')
    elif chatglm_style:
        return parsed_context
    else:
        # Deprecated
        return f"{context}\n\nGiven the dialogue above, write a response.\n"
    
def _build_rolechat_prompt(context: Union[List[str], str], init_prompt, inline_prompt, chatglm_style=False, plug_style=False, merge=False, dialog_sep='\n'):
    dialogs_with_prompt = _build_chitchat_prompt(context, not chatglm_style, chatglm_style, dialog_sep=dialog_sep)
    if merge:
        return f"{init_prompt + inline_prompt.strip('：:')}\n\n{dialogs_with_prompt}"
    
    if dialogs_with_prompt.endswith(p2_prompt):
        dialogs_with_prompt = dialogs_with_prompt[:-len(p2_prompt)]
    elif dialogs_with_prompt.endswith(p1_prompt):
        dialogs_with_prompt = dialogs_with_prompt[:-len(p1_prompt)]
    
    if plug_style:
        return f"{init_prompt.replace('{context}', dialogs_with_prompt, 1)}你的回复是："
    return f"{init_prompt}\n\n{dialogs_with_prompt}{dialog_sep}{inline_prompt}"

# def set_optim_to_run_embedding_in_fp32(model: torch.nn.Module):
#     from bitsandbytes.optim import GlobalOptimManager
#     for module in model.modules():
#         if isinstance(module, torch.nn.Embedding):
#             GlobalOptimManager.get_instance().register_module_override(module, 'weight', {'optim_bits': 32})
            
def pad_sequences(lst_seq: List[List[int]], pad_value, pad_left=False, pad_to: int=None) -> List[List[int]]:
    maxlen = max(len(seq) for seq in lst_seq) if pad_to is None else pad_to
    if pad_left:
        padded_seq = [[pad_value] * (maxlen - len(seq)) + seq for seq in lst_seq]
    else:
        padded_seq = [seq + [pad_value] * (maxlen - len(seq)) for seq in lst_seq]
    return padded_seq

def simple_retrieve(kv_pairs, s):
    return [v for k, v in kv_pairs.items() if k in s]

class History():
    def __init__(self, opt) -> None:
        self.delimiter: str = opt.delimiter
        self.context: List[str] = []
    
    def add(self, sent):
        self.context.append(sent)
        
    def reset(self, keep_last=False):
        if keep_last:
            self.context = self.context[-1:]
        else:
            self.context.clear()
        
    def get_full_context(self, as_str=True) -> Union[str, List[str]]:
        if as_str:
            return self.delimiter.join(self.context)
        return self.context.copy()
