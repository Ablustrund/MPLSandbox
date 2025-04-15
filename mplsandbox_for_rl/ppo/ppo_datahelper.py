from typing import Dict, Any, List, Tuple, Union, Generator
import json, logging, torch, copy
from data_helper import DialogDataset
from dataclasses import dataclass
from utils import get_separate_prompt, build_prompt, pad_sequences, build_code_context
import random

@dataclass
class Cell:
    state: str
    score: int
    visit: int = 0

class Archive():
    def __init__(self):
        self.items : Dict[Any,Cell] = dict()
        self.count = 0
        self.floor_score = float('inf')
        self.floor_key = None
        
    def add(self, state: str, score: float):
        key = hash(state)
        self.items[key] = Cell(state, score)
        self.count += 1
        if score < self.floor_score:
            self.floor_score = score
            self.floor_key = key

    def update(self, state: str, score: float):
        k = hash(state)
        if k in self.items.keys():
            if score > self.items[k].score:
                self.items[k].score = score
            elif score == self.items[k].score and len(self.items[k].state) > len(state):
                self.items[k].state = state
        else:
            self.items[k] = Cell(state, score)
            del self.items[self.floor_key]
            scores = [i.score for i in self.items]
            keys = [key for key, _ in self.items]
            self.floor_score = min(scores)
            self.floor_key = keys[scores.index(self.floor_score)]

# archives = [Archive() for i in range(16)]

class DialogPromptDataset(DialogDataset):
    def __init__(self, opt, accelerator, mode: str = 'train', **kwargs) -> None:
        super().__init__(opt, accelerator, mode, **kwargs)
        # global archives
        self.dynamic_batching = False
        self.batch_size = opt.rollout_batch_size
        self.max_ts = opt.max_ts
    
        self.archives = [Archive() for i in range(16)]

    
    def _load_data(self, dpath: str):
        with open(dpath, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
            
        output: List[Dict[str, Any]] = [sample for sample in data if all(sample)]
        error_samples = [sample for sample in data if not all(sample)]
            
        if error_samples:
            logging.warn(f'Detected {len(error_samples)} illegal samples')
            logging.warn(f'Examples: {error_samples[:5]}')
            
        del data, error_samples
        return output

    def _encode_sample(self, sample:Dict[str, Any]) -> Dict[str, Any]:
        
        # print("Address in prompt dataset:", id(self.archives))
        # for a in self.archives:
        #     print(id(a.items))
        #     print(a.count)
        code_description = sample['prompt']
        golden_solution = sample['canonical_solution']
        inputs = sample['inputs']
        outputs = sample['outputs']
        starter_code = sample['starter_code']
        difficulty = sample['difficulty']
        index = sample['id']
        start_state = ""

        prompt = build_prompt(code_description, starter_code, openai_style=self.opt.openai_style_prompt, dialog_sep=self.opt.delimiter)
        if self.mode == 'train':
            # 改一下prompt构造，从archive[i]里random select（权重）或者不select（概率）选state
            count = self.archives[index].count
            # print(f"count:{count}")
            # 从archive里选
            if random.random() < self.opt.random_ratio and count > 0:
                items = self.archives[index].items
                def weighted_choice(items, count):
                    visit_counts = [items[i].visit for i in items.keys()]
                    weighted_probs = [1 / (0.5 * count + 1) for count in visit_counts]
                    total_weight = sum(weighted_probs)
                    normalized_probs = [weight / total_weight for weight in weighted_probs]
                    return random.choices(list(range(count)), weights=normalized_probs, k=1)[0]

                selected_id = weighted_choice(items, count)
                items[selected_id].visit += 1
                start_state = items[selected_id].state
            prompt += start_state

        context_vec = self.tokenizer.txt2vec(prompt)

        if len(context_vec) > self.c_trunc - self.max_ts:
            context_vec = context_vec[-(self.c_trunc - self.max_ts):]
        
        
        text_len = len(context_vec)

        output = {
            'id': index,
            'text_vec': context_vec,
            'text': self.tokenizer.vec2txt(context_vec),
            'text_len': text_len,
            'golden_solution': golden_solution,
            'inputs': inputs,
            'outputs': outputs,
            'starter_code': starter_code,
            'start_state': start_state,
            'difficulty': difficulty,
        }
        
        return output
    
    def _get_allowed_max_len(self):
        return 999999
    
    def _get_sample_len(self, sample: Dict[str, Any]):
        return len(sample['text_vec'])
    
    def batch_generator(self) -> Generator[List[Dict[str, Any]], None, None]:
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size: # drop last
                    yield batch
            if self.mode != 'train':
                break
                
    def dynamic_batch_generator(self) -> Generator[List[Dict[str, Any]], None, None]:
        return self.batch_generator() # dynamic batching is not available for PPO
    
    def _batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_text_vec = torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.null_token_id, pad_left=True), dtype=torch.long)
        return {
            'ids': [sample['id'] for sample in batch_samples],
            'text_vec': batch_text_vec,
            'text': [sample['text'] for sample in batch_samples],
            'text_len': [sample['text_len'] for sample in batch_samples],
            'golden_solution': [sample['golden_solution'] for sample in batch_samples],
            'inputs': [sample['inputs'] for sample in batch_samples],
            'outputs': [sample['outputs'] for sample in batch_samples],
            'starter_code': [sample['starter_code'] for sample in batch_samples],
            'start_state': [sample['start_state'] for sample in batch_samples],
            'difficulty': [sample['difficulty'] for sample in batch_samples],
            'text_trunc': [1 if sample['text_len'] > len(sample['text_vec']) else 0 for sample in batch_samples],
            'n_tokens': sum(len(sample['text_vec']) for sample in batch_samples),
        }
        
class DialogReplayDataset(DialogDataset):
    def __init__(self, data: List[Dict[str, Any]], opt, accelerator, archives, mode: str = 'train', **kwargs) -> None:
        super(DialogDataset, self).__init__(opt, accelerator, mode, **kwargs)
        self.gamma = opt.gamma
        self.lam = opt.lam
        self.data = data
        self.size = len(data)
        self.archives = archives
        if self.accelerator.use_distributed:
            self.size *= self.accelerator.num_processes
    
    def _load_data(self, dpath: str):
        return []
    
    def _get_advantages_and_returns(self, rewards: List[float], values: List[float]):
        '''
        Copied from TRLX: https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        '''
        response_length = len(values)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(response_length)):
            nextvalues = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + self.gamma * nextvalues - values[t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            
        advantages = advantages_reversed[::-1]
        returns = [a + v for a, v in zip(advantages, values)]
        assert len(returns) == len(advantages) == len(values)
        return advantages, returns
    
    def _encode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        output = copy.deepcopy(sample)
        advantages, returns = self._get_advantages_and_returns(sample['reward'], sample['values'])

        max_return = max(returns)
        max_return_pos = returns.index(max_return)
        start_state = sample['start_state']
        max_return_state = start_state + sample['resp'][:max_return_pos+1]

        index = sample['id']
        archive = self.archives[index]
        items = archive.items

        if archive.count < self.opt.archive_size:
            # 没满直接加
            self.archives[index].add(max_return_state, max_return)
        else:
            # 满了只有更大的return的情况下才能update
            if max_return > archive.floor_score:
                self.archives[index].update(max_return_state, max_return)
        
        print("Address in replay dataset:", id(self.archives))
        for a in self.archives:
            print(id(a.items))
            print(a.count)
        context_vec, resp_vec = sample['context_vec'], sample['resp_vec']
        assert len(resp_vec) == len(advantages) == len(returns)
        
        text_vec = context_vec + resp_vec
        loss_mask = [0] * len(context_vec) + [1] * len(resp_vec)
        
        output['text_vec'] = text_vec
        output['text'] = self.tokenizer.vec2txt(text_vec)
        output['text_len'] = len(text_vec)
        output['label_len'] = len(resp_vec)
        output['advantages'] = [0.] * (len(context_vec) - 1) + advantages
        output['returns'] = [0.] * (len(context_vec) - 1) + returns
        output['values'] = [0.] * (len(context_vec) - 1) + output['values']
        output['logprobs'] = [0.] * (len(context_vec) - 1) + output['logprobs']
        output['loss_mask'] = loss_mask
        return output
    
    def _get_allowed_max_len(self):
        return 999999
    
    def _get_sample_len(self, sample: Dict[str, Any]):
        return len(sample['text_vec'])
    
    def batch_generator(self) -> Generator[List[Dict[str, Any]], None, None]:
        for batch in super().batch_generator():
            yield batch
                    
    def dynamic_batch_generator(self) -> Generator[List[Dict[str, Any]], None, None]:
        return self.batch_generator()
    
    def _batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            'text': [sample['text'] for sample in batch_samples],
            'context': [sample['context'] for sample in batch_samples],
            'resp': [sample['context'] for sample in batch_samples],
            'text_vec': torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.null_token_id), dtype=torch.long),
            'loss_mask': torch.tensor(pad_sequences([sample['loss_mask'] for sample in batch_samples], pad_value=0), dtype=torch.bool),
            
            'advantages': torch.tensor(pad_sequences([sample['advantages'] for sample in batch_samples], pad_value=0.)),
            'returns': torch.tensor(pad_sequences([sample['returns'] for sample in batch_samples], pad_value=0.)),
            'values': torch.tensor(pad_sequences([sample['values'] for sample in batch_samples], pad_value=0.)),
            'logprobs': torch.tensor(pad_sequences([sample['logprobs'] for sample in batch_samples], pad_value=0.)),
            
            'text_len': [sample['text_len'] for sample in batch_samples],
            'label_len': [sample['label_len'] for sample in batch_samples],
            'text_trunc': [1 if sample['text_len'] > len(sample['text_vec']) else 0 for sample in batch_samples],
            'n_tokens': sum(len(sample['text_vec']) for sample in batch_samples),
        }
        return batch