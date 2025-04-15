import torch
import uuid, os, re, subprocess
import math, time, logging
from typing import Dict, Any, Union, List, Tuple
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from tokenizer import HFPretrainedTokenizer as LlamaPretrainedTokenizer
from utils import pad_sequences, build_code_context
from datetime import datetime
from mplsandbox import MPLSANDBOX


class LlamaCriticModel(LlamaForCausalLM):
    def __init__(self, config, opt: Dict[str, Any], dict: LlamaPretrainedTokenizer, **kwargs):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = dict
        
        self.debug_mode: bool = opt.debug
        self.end_idx = dict.end_token_id
        self.NULL_IDX = dict.null_token_id
        
        self.reward_head = torch.nn.Linear(config.hidden_size, 1, bias=False)
        self.calculate_lm_loss: bool = getattr(opt, 'reward_lm_loss_factor', 0.) > 0.
        self.post_init()
        
    def forward(self, decoder_input: torch.LongTensor, rank_all=False):
        if not (rank_all or decoder_input[:, -1].eq(self.end_idx).all()):
            logging.warn(f'Found sample that NOT ended with EOS token')
        
        attention_mask = decoder_input.ne(self.NULL_IDX)
        output = self.model.forward(input_ids=decoder_input, attention_mask=attention_mask, 
                                          return_dict=True, use_cache=False)
        if not rank_all:
            logits = output.last_hidden_state[:, -1, :]
            logits = self.reward_head(logits).squeeze(-1)
        else:
            logits = self.reward_head(output.last_hidden_state).squeeze(-1)
            
        if self.calculate_lm_loss:
            lm_logits = self.lm_head(output.last_hidden_state)
            return logits, lm_logits
        
        return (logits,)
    
    def reward(self, context: Union[str, List[str], List[int]], is_encoded=False, **kwargs):
        text_vec = self._preprocess_context(context) if not is_encoded else context
        text_vec = torch.tensor([text_vec], dtype=torch.long, device=self.device)
        score = self.forward(text_vec, **kwargs)[0].squeeze(0)
        return score.cpu()
        
    
    def batch_reward(self, context: Union[List[str], List[List[str]], List[List[int]]], is_encoded=False, **kwargs):
        text_vec = [self._preprocess_context(s) for s in context] if not is_encoded else context
        text_vec = pad_sequences(text_vec, pad_value=self.NULL_IDX, pad_left=True)
        text_vec = torch.tensor(text_vec, dtype=torch.long, device=self.device)
        score = self.forward(text_vec, **kwargs)[0]
        return score.cpu()

class LlamaRewardModel():
    def __init__(self, opt: Dict[str, Any], dict: LlamaPretrainedTokenizer, **kwargs):
        self.opt = opt
        self.tokenizer = dict
        
        self.debug_mode: bool = opt.debug
        self.end_idx = dict.end_token_id
        self.NULL_IDX = dict.null_token_id
    
    def process_answer(self, text):
        text = text.strip()
        
        if '```' in text:
            blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                # if not text.startswith('\n'):  # in case starting with ```python
                #     text = text[max(text.find('\n') + 1, 0):]
        else:
            match = re.search(r'Here(.*?)\n', text)
            if match:
                text = re.sub('Here(.*?)\n', '', text, count=1)
            match = re.search(r'One approach(.*?)\n', text)
            if match:
                text = re.sub('One approach(.*?)\n', '', text, count=1)

        return text.strip('</s>')

    def write_tmp(self, code, tmp_file):
        def insert_library(code):
            # Using a single multi-line string to reduce concatenations
            libs = """
import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations, groupby
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import Dict, Any, List, Optional, Tuple, Union, Generator
import numpy as np
import random
import heapq as hq
import re
from heapq import *
import bisect
import resource

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, maxsize+1))
limit_memory(524288000)
    """
            return libs + code

        code = insert_library(code)
        file = open(tmp_file, 'w')
        file.write(code)
        file.close()

    def find_test(self, code):
        method_pos = code.find("def ") + 4
        method_name = code[method_pos : code.find("(", method_pos)]
        begin_idx =  method_pos + len(method_name)
        
        test_pos = code.find(method_name, begin_idx)
        substrings_set = ["# Output:","# Example","# Sample","# Test", "### Note","# Explanation:","if __name__ == \"__main__\":","if __name__ == '__main__':",method_name + "()",f"print({method_name}("]
        idx = -1
        for substring in substrings_set:
            index = code.find(substring, begin_idx)
            if index != -1 and idx != -1:
                idx = min(idx, index)
            elif idx == -1:
                idx = index
        return idx

    def find_test_type(self, code):
        method_pos = code.find("def ") + 4
        method_name = code[method_pos : code.find("(", method_pos)]
        begin_idx =  method_pos + len(method_name)
        
        test_pos = code.find(method_name, begin_idx)
        # 没有自己写测试
        if test_pos == -1:
            return -1
        else:
            output_pos = code.find("# Output:", test_pos)
            # 自己写了输出结果 是return類型 得assert 當作是沒有自己寫測試
            if output_pos != -1:
                return 0
            # 在函数外捕获输入 不需要自己加
            else:
                return 1
        

    def execute_code(self, code, inputs, outputs, tmp_file,has_args, has_print):
        breakpoint()
        self.write_tmp(code,tmp_file)
        # 构建命令
        command = f"python3 \"{tmp_file}\""
        if not has_args:
            command = f"echo \"{inputs}\" | " + command

        # 执行命令并捕获输出和错误
        try:
            process = subprocess.run(command, shell=True, capture_output=True, timeout=3, text=True)
            output = process.stdout
            error = process.stderr

            # 分析错误和输出
            if "SyntaxError" in error:
                return -1, error
            elif "AssertionError" in error:
                return -0.3, error
            elif error != "":
                return -0.6, error
            else:
                if has_print:
                    output = [i.strip() for i in output.strip("\n").split("\n")]
                    if isinstance(outputs, str):
                        expected_outputs = [i.strip() for i in outputs.strip("\n").split("\n")]
                    elif isinstance(outputs, list):
                        expected_outputs = outputs
                    if output != expected_outputs:
                        return -0.3, f"AssertionError:\n{output}\n{expected_outputs}"
                return 1, "unit pass"
        except subprocess.TimeoutExpired:
            return -0.6, "time out"
        

    def get_reward(self, resp, inputs, outputs, tmp_file):
        try:
            
            code = self.process_answer(resp)
            method_pos = code.find("def ") + 4
            method_name = code[method_pos : code.find("(", method_pos)]
            # method_name = code[code.index("def ") + 4:code.find("(", code.index("def ") + 4)]
            # begin_idx = code.index("def " + method_name) + len("def " + method_name)
            test_type = self.find_test_type(code)
            breakpoint()
            # 需要自己加單元測試語句的情況
            if test_type != 1:
                if test_type == 0:
                    test_pos = self.find_test(code)
                    def remove_lines_after_index(input_string, index):
                        lines = input_string.split('\n')
                        total_chars = 0

                        for i, line in enumerate(lines):
                            total_chars += len(line) + 1 

                            if total_chars > index:
                                lines = lines[:i]
                                break

                        return '\n'.join(lines)
                    code = remove_lines_after_index(code, test_pos)

                in_class = 'class Solution' in code
                has_args = code[code.find("(", code.index("def " + method_name)) + 1] != ')'
                print_index = max(code.rfind("print"), code.rfind("stdout.write"), code.rfind("stdout.write_array"))
                has_print = True if code.rfind("print") > code.rfind(method_name) else False
                # has_return = True if code.rfind("return") > print_index else False
                # check if the method contains arguments

                if in_class:
                    code += "\nsolution = Solution()"
                    # code += "\nSolution()" 
                
                ori_code = code
                # print(starter_code)
                for unit_test in zip(inputs, outputs):
                    code = ori_code
                    u_input, u_output = unit_test
                    method_call = f"{method_name}()" if not has_args else f"{method_name}({','.join(map(str, u_input))})"

                    if not has_print:
                        solution_prefix = "\nassert solution." if in_class else "\nassert "
                        outputs_str = ""
                        if isinstance(u_output, str):
                            ans = u_output.strip('\n').split('\n')
                            if len(ans) == 1:
                                outputs_str = str(ans[0])
                            else:
                                outputs_str = str(ans)
                        else:
                            outputs_str = ",".join(map(str, u_output))
                        code += solution_prefix + method_call + " == " + str(outputs_str)
                    else:
                        solution_prefix = "\nsolution." if in_class else "\n"
                        code += solution_prefix + method_call
                    breakpoint()
                    reward, error = self.execute_code(code, u_input, u_output, tmp_file, has_args, has_print)
                    if reward != 1:
                        return reward, error
                return 1, "all pass"

            else:
                for unit_test in zip(inputs, outputs):
                    reward, error = self.execute_code(code, u_input, u_output, tmp_file, has_args=False, has_print=True)
                    if reward != 1:
                        return reward, error
                return 1, "all pass"
        except Exception as e:
            return 0, f"An error occurred: {e}"
        

    def get_reward_via_mplsandbox(self, resp, inputs, outputs, tmp_file):
        try:
            
            code = self.process_answer(resp)
            method_pos = code.find("def ") + 4
            method_name = code[method_pos : code.find("(", method_pos)]
            # method_name = code[code.index("def ") + 4:code.find("(", code.index("def ") + 4)]
            # begin_idx = code.index("def " + method_name) + len("def " + method_name)
            test_type = self.find_test_type(code)

            if test_type != 1:
                if test_type == 0:
                    test_pos = self.find_test(code)
                    def remove_lines_after_index(input_string, index):
                        lines = input_string.split('\n')
                        total_chars = 0

                        for i, line in enumerate(lines):
                            total_chars += len(line) + 1 

                            if total_chars > index:
                                lines = lines[:i]
                                break

                        return '\n'.join(lines)
                    code = remove_lines_after_index(code, test_pos)

                in_class = 'class Solution' in code
                has_args = code[code.find("(", code.index("def " + method_name)) + 1] != ')'
                print_index = max(code.rfind("print"), code.rfind("stdout.write"), code.rfind("stdout.write_array"))
                has_print = True if code.rfind("print") > code.rfind(method_name) else False
                # has_return = True if code.rfind("return") > print_index else False
                # check if the method contains arguments

                if in_class:
                    # code += "\nsolution = Solution()"
                    code += "\nSolution()" 
                
                ori_code = code
                code = ori_code
                method_call = f"{method_name}()" if not has_args else f"{method_name}({','.join(map(str, u_input))})"

                if not has_print:
                    solution_prefix = "\nres = "
                    code += solution_prefix + method_call + "\nprint(res)" 
                else:
                    solution_prefix = "\nsolution." if in_class else "\n"
                    code += solution_prefix + method_call                
                data_for_sandbox = {"question":"---","code":code,"unit_cases": {"inputs": inputs,"outputs": outputs},"lang":"python"}
                executor = MPLSANDBOX(data_for_sandbox)
                results = executor.get_basic_info()
                reward, compiler_feedback = results['reward'], results['compiler_feedback']
                if reward != 1:
                    return reward, compiler_feedback
                else:
                    return 1, "all pass"

            else:
                data_for_sandbox = {"question":"---","code":code,"unit_cases": {"inputs": inputs,"outputs": outputs},"lang":"python"}
                executor = MPLSANDBOX(data_for_sandbox)
                results = executor.get_basic_info()
                reward, compiler_feedback = results['reward'], results['compiler_feedback']
                if reward != 1:
                    return reward, compiler_feedback
                else:
                    return 1, "all pass"
        except Exception as e:
            return 0, f"An error occurred: {e}"
        

    def forward(self, resp_vec_sampled: List[List[int]], resps: List[str], batch, bsz: int, mode='train'):
        code_descriptions: List[str] = batch['text']
        golden_solutions: List[str] = batch['golden_solution']
        batch_inputs = batch['inputs']
        batch_outputs = batch['outputs']
        starter_code = batch['starter_code']
        start_state = batch['start_state']
        difficulty = batch['difficulty']
               
        rewards = []
        for i in range(bsz):
            currentDateAndTime = datetime.now()
            
            log_dir = f"tmp/{self.opt.model_file.split('/')[-1]}/code/{mode}/"
            if mode == 'test':
                log_dir = f"tmp/test/{self.opt.model_file.split('/')[-1]}/{difficulty[i]}"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            tmp_file = f'{log_dir}/{currentDateAndTime}.py'
            start_flag = '# continue...\n'
            

            reward,result = self.get_reward_via_mplsandbox(f"```python\n{starter_code[i]}\n" + start_state[i] + start_flag + resps[i], batch_inputs[i], batch_outputs[i], tmp_file)
            file = open(tmp_file, 'a')
            file.write(f"\n# --------reward------------\n'''\n{reward}\n'''\n")
            file.write(f"\n# --------result------------\n'''\n{result}\n'''\n")
            file.write(f"\n# --------difficulty------------\n'''\n{difficulty[i]}\n'''\n")
            file.write(f"\n\n# --------instruction------------\n'''\n{code_descriptions[i]}\n'''\n")
            file.write(f"\n# --------response------------\n'''\n{resps[i]}\n'''\n")
            file.write(f"\n# --------inputs------------\n'''\n{batch_inputs[i]}\n'''\n")
            file.write(f"\n# --------outputs------------\n'''\n{batch_outputs[i]}\n'''\n")
            file.write(f"\n# --------golden------------\n'''\n{golden_solutions[i]}\n'''\n")
            file.close()
            rewards.append(reward)
               
        ones = [1 if x == 1. else 0 for x in rewards]
        rewards = torch.tensor(rewards, dtype=torch.float16)
        introductory = [1 if diff=="introductory" and rewards[i] == 1. else 0 for i, diff in enumerate(difficulty)]
        # print(introductory)
        interview = [1 if diff=="interview" and rewards[i] == 1. else 0 for i, diff in enumerate(difficulty)]
        competition = [1 if diff=="competition" and rewards[i] == 1. else 0 for i, diff in enumerate(difficulty)]
        print(f"rewards is {rewards}")
        return rewards, ones, introductory, interview, competition
        

if __name__ == '__main__':
    rewardmodel = LlamaRewardModel(None, None)
