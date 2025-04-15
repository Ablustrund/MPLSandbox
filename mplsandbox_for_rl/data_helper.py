from argparse import Namespace
from accelerate import Accelerator
from itertools import repeat
from typing import Dict, Any, List, Optional, Tuple, Union, Generator
import logging
from collections import deque
import torch
import os, math, json, random
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from tokenizer import HFPretrainedTokenizer
from utils import *

class DummyDataset(IterableDataset):
    def __iter__(self):
        for sample in repeat(None):
            yield sample
    
    def __len__(self):
        return 1024

class BaseDataset(IterableDataset):
    def __init__(self, opt: Namespace, accelerator: Accelerator, mode: str='train', **kwargs) -> None:
        super().__init__()
        assert mode in ('train', 'valid', 'test')
        self.opt = opt
        self.accelerator = accelerator
        
        self.mode = mode
        self.batch_size = opt.batch_size
        self.debug: bool = opt.debug
        self.stoppable: bool = kwargs.get('stoppable', False) # iterate 1 epoch only for train mode instead of infinite loop
        self.verbose: int = opt.verbose
        # self.no_label = no_label
        # if no_label:
        #     assert mode == 'test', 'Labels are required for training and evaluation'
            
        if mode in ('valid', 'test') and not opt.skip_generation:
            self.generation_mode = True
            self.max_ts = opt.max_ts
        else:
            self.generation_mode = False
            
        self.tokenizer = self.tokenizer_class()(opt)
        self.c_trunc: int = opt.context_truncate
        self.dynamic_batching: bool = opt.dynamic_batching
        self.no_split_dialog: bool = opt.no_split_dialog
        if opt.chatglm_style_prompt:
            assert not self.no_split_dialog, 'GLM model does NOT support `no_split_dialog`, dialogs should already split before'
            
    def tokenizer_class(self):
        return HFPretrainedTokenizer

    def _load_data(self, dpath: str):
        with open(dpath, 'r') as f:
            data: List[List[str]] = json.load(f)

        output: List[Tuple[List[str], str]] = []
        error_samples = []
        for turn in data:
            if not isinstance(turn, list) or len(turn) < 2 or not all(turn):
                error_samples.append(turn)
                continue
            # if self.no_split_dialog and len(turn) % 2 == 1:
            #     error_samples.append(turn)
            #     continue
            
            if self.no_split_dialog:
                output.append((turn, ''))
            else:
                output.append((turn[:-1], turn[-1]))
        
        if error_samples:
            logging.warn(f'Detected {len(error_samples)} illegal samples')
            logging.warn(f'Examples: {error_samples[:5]}')

        del data, error_samples
        return output
    
    def _get_bot_mask(self, text_vec: List[int], sep_i: int):
        mask, cnt = [], 0
        for v in text_vec:
            mask.append(cnt % 2)
            cnt += int(v == sep_i)
        return mask
    
    def _encode_sample(self, sample: Tuple[List[str], str]) -> Dict[str, Any]:
        # tokenize text
        context, label = sample
        if self.opt.chatglm_style_prompt and len(context) % 2 == 0:
            context = context[1:]
        context = [get_separate_prompt(i) + u for i, u in enumerate(context)]
        
        label_vec = (self.tokenizer.txt2vec(label) + [self.tokenizer.end_token_id]) if not self.no_split_dialog else []
        if not self.generation_mode:
            length_limit = self.c_trunc
        else:
            length_limit = self.c_trunc - self.max_ts
            
        # tokenize text
        if self.no_split_dialog:
            context_vec = self.tokenizer.txt2vec(build_prompt(self.tokenizer.end_token.join(context), 
                                                              openai_style=self.opt.openai_style_prompt, 
                                                              dialog_sep=self.tokenizer.end_token))
        else:
            context_vec = self.tokenizer.txt2vec(build_prompt(context, 
                                                              openai_style=self.opt.openai_style_prompt, 
                                                              chatglm_style=self.opt.chatglm_style_prompt, 
                                                              dialog_sep=self.opt.delimiter))
            if self.opt.chatglm_style_prompt:
                context_vec += [self.tokenizer.gmask_token_id, self.tokenizer.start_token_id]
            
        text_len = len(context_vec) + len(label_vec)
        
        # truncate text
        if self.no_split_dialog:
            context_vec = context_vec[:length_limit]
        else:
            while len(context_vec) + len(label_vec) > length_limit and len(context) > 1:
                context = context[2 if self.opt.chatglm_style_prompt else 1:]
                context_vec = self.tokenizer.txt2vec(build_prompt(context, 
                                                                  openai_style=self.opt.openai_style_prompt, 
                                                                  chatglm_style=self.opt.chatglm_style_prompt, 
                                                                  dialog_sep=self.opt.delimiter))
                if self.opt.chatglm_style_prompt:
                    context_vec += [self.tokenizer.gmask_token_id, self.tokenizer.start_token_id]
                
        text_vec = context_vec + label_vec
        if self.no_split_dialog:
            loss_mask = self._get_bot_mask(text_vec, sep_i=self.tokenizer.end_token_id)
            label_start_pos = 1
        else:
            loss_mask = [0] * len(context_vec) + [1] * len(label_vec)
            label_start_pos = len(context_vec)
            assert text_vec[label_start_pos:] == label_vec
        if self.debug:
            print(f'Non-masked part: {self.tokenizer.vec2txt([tok for tok, i in zip(text_vec, loss_mask) if i > 0])}')

        output = {
            'text': sample,
            'text_encoded': self.tokenizer.vec2txt(text_vec),
            'text_vec': text_vec,
            'text_len': text_len,
            'loss_mask': loss_mask,
            'label_pos': label_start_pos,
            'label': label,
        }
        if self.no_split_dialog:
            del output['label_pos']
        return output

    def _get_sample_len(self, sample: Dict[str, Any]):
        return len(sample['text_vec'])
    
    def _get_allowed_max_len(self):
        return self.c_trunc
    
    def _batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # create a batch from a list of samples
        batch = dict()
        batch_text_vec = torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.null_token_id, pad_left=self.generation_mode), dtype=torch.long)
        loss_mask = torch.tensor(pad_sequences([sample['loss_mask'] for sample in batch_samples], pad_value=0, pad_left=self.generation_mode), dtype=torch.bool)
        if self.generation_mode:
            batch_text_vec_no_label = torch.tensor(pad_sequences([sample['text_vec'][:sample.get('label_pos', len(sample['text_vec']))] for sample in batch_samples], pad_value=self.tokenizer.null_token_id, pad_left=self.generation_mode), dtype=torch.long)
        
        batch.update({
            'text_vec': batch_text_vec,
            'loss_mask': loss_mask,
            'text_len': [sample['text_len'] for sample in batch_samples],
            'n_tokens': sum(len(sample['text_vec']) for sample in batch_samples),
            'text_trunc': [1 if sample['text_len'] > len(sample['text_vec']) else 0 for sample in batch_samples],
            'label_pos': min(sample.get('label_pos', 1) for sample in batch_samples) if not self.generation_mode else 1,
        })
        
        # for debug only
        batch.update({
            'text': [sample['text'] for sample in batch_samples],
            'text_encoded': [sample.get('text_encoded', None) for sample in batch_samples],
        })
        
        # for generation mode
        if self.generation_mode:
            batch['label'] = [sample.get('label', 'NO LABEL') for sample in batch_samples]
            batch['text_vec_no_label'] = batch_text_vec_no_label
        
        return batch

    def sample_generator(self) -> Generator[Dict[str, Any], None, None]:
        # generate one sample per time
        # handled by sub-classes
        raise NotImplementedError
    
    def batch_generator(self) -> Generator[List[Dict[str, Any]], None, None]:
        # generate batches with fixed size
        max_len = self._get_allowed_max_len()
        min_len = 1
        batch: List[Dict[str, Any]] = []

        for sample in self.sample_generator():
            # skip illegal samples
            sample_len = self._get_sample_len(sample)
            if not (min_len <= sample_len <= max_len):
                if self.verbose > 0:
                    logging.warn(f'Found sample with length of {sample_len} which > {max_len} or < {min_len}, skipped')
                continue
            
            # add legal samples to batch
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch
    
    def dynamic_batch_generator(self) -> Generator[List[Dict[str, Any]], None, None]:
        # generate dynamic batches with fixed num of tokens
        max_tokens = self.batch_size * self._get_allowed_max_len()
        num_buckets = 128
        max_len = self._get_allowed_max_len()
        min_len = 1

        buckets: List[List[Any]] = [[] for _ in range(num_buckets)]
        buckets_maxlen: List[int] = [0] * num_buckets

        def is_batch_full(num_tokens, batch):
            if len(batch) == 0:
                return False
            if num_tokens > max_tokens:
                return True
            return False

        for sample in self.sample_generator():
            sample_len = self._get_sample_len(sample)
            # skip illegal samples
            if not (min_len <= sample_len <= max_len):
                if self.verbose > 0:
                    logging.warn(f'Found sample with length of {sample_len} which > {max_len} or < {min_len}, skipped')
                    # print(sample)
                continue

            index_buckets = math.floor((sample_len - min_len) / (max_len - min_len + 1) * num_buckets)
            index_buckets = min(index_buckets, num_buckets - 1) # to avoid out of index
            
            buckets_maxlen[index_buckets] = max(buckets_maxlen[index_buckets], sample_len)
            num_tokens = (len(buckets[index_buckets]) + 1) * buckets_maxlen[index_buckets]
            if is_batch_full(num_tokens, buckets[index_buckets]):
                # get a full batch, yield it
                yield buckets[index_buckets]
                buckets[index_buckets] = []
                buckets_maxlen[index_buckets] = sample_len
            # add current sample to a bucket
            buckets[index_buckets].append(sample)
        
        # process left-over
        leftover_batch = []
        leftover_maxlen = 0
        leftover = [sample for bucket in buckets for sample in bucket]
        for sample in leftover:
            sample_len = self._get_sample_len(sample)
            # skip illegal samples
            if not (min_len <= sample_len <= max_len):
                logging.warn(f'Found sample with length of {sample_len} which > {max_len} or < {min_len}, skipped')
                continue

            leftover_maxlen = max(leftover_maxlen, sample_len)
            num_tokens = (len(leftover_batch) + 1) * leftover_maxlen
            if is_batch_full(num_tokens, leftover_batch):
                # get a full batch, yield it
                yield leftover_batch
                leftover_batch = []
                leftover_maxlen = sample_len
            # add current sample to bucket
            leftover_batch.append(sample)

        # last batch
        if leftover_batch:
            yield leftover_batch

    def final_generator(self) -> Generator[Dict[str, Any], None, None]:
        # generate padded batch from batch_generator or dynamic_batch_generator
        if self.dynamic_batching:
            data_generator = self.dynamic_batch_generator()
        else:
            data_generator = self.batch_generator()

        for batch_samples in data_generator:
            batch = self._batchify(batch_samples)
            yield batch

    def __iter__(self):
        return self.final_generator()

class DialogDataset(BaseDataset):
    # For datasets which can fit in the memory
    # Use ChunkDataset otherwise
    def __init__(self, opt, accelerator, mode: str='train', **kwargs) -> None:
        super().__init__(opt, accelerator, mode, **kwargs)

        # load data, support for suffix loading
        self.data = []
        fpaths = sorted([f_name for f_name in os.listdir(opt.data_path) if f_name.endswith(f'{mode}.json')])
        for f_name in fpaths:
            # if f_name.endswith(f'{mode}.json'):
            dpath = os.path.join(opt.data_path, f_name)
            data_dpath = []
            try:
                data_dpath = self._load_data(dpath)
            except Exception as e:
                logging.warn(f"Load data from {dpath} failed. {str(e)}")
            self.data.extend(data_dpath)
            logging.info(f'Got {len(data_dpath)} samples from {dpath}')
        logging.info(f'Got {len(self.data)} samples totally from {fpaths}')
        # load data for single file
        # dpath = os.path.join(opt.data_path, f'{mode}.json')
        # self.data = self._load_data(dpath)

        self.size = len(self.data)

        # get data for current rank on distributed train
        if accelerator and self.accelerator.use_distributed:
            self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]
        # logging.info(f'Got {len(self.data)} samples from {dpath}')

    def sample_generator(self):
        random.seed(None)
        need_shuffle = self.mode == 'train'
        
        # If multiprocessing dataloader is used, set data for current worker
        worker_info = get_worker_info()
        if worker_info is not None:
            self.data = self.data[worker_info.id::worker_info.num_workers]
            logging.info(f'WORKER {worker_info.id} Got {len(self.data)} samples')
            
        if need_shuffle:
            random.shuffle(self.data)
        # yield samples
        for sample in self.data:
            yield self._encode_sample(sample)
            
    def __len__(self):
        return self.size

class CodeDataset(BaseDataset):
    # For datasets which can fit in the memory
    # Use ChunkDataset otherwise
    def __init__(self, opt, accelerator, mode: str='train', **kwargs) -> None:
        super().__init__(opt, accelerator, mode, **kwargs)

        # load data, support for suffix loading
        self.data = []
        fpaths = sorted([f_name for f_name in os.listdir(opt.data_path) if f_name.endswith(f'{mode}.json')])
        for f_name in fpaths:
            # if f_name.endswith(f'{mode}.json'):
            dpath = os.path.join(opt.data_path, f_name)
            data_dpath = []
            try:
                data_dpath = self._load_data(dpath)
            except Exception as e:
                logging.warn(f"Load data from {dpath} failed. {str(e)}")
            self.data.extend(data_dpath)
            logging.info(f'Got {len(data_dpath)} samples from {dpath}')
        logging.info(f'Got {len(self.data)} samples totally from {fpaths}')
        # load data for single file
        # dpath = os.path.join(opt.data_path, f'{mode}.json')
        # self.data = self._load_data(dpath)

        self.size = len(self.data)

        # get data for current rank on distributed train
        if accelerator and self.accelerator.use_distributed:
            self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]
        # logging.info(f'Got {len(self.data)} samples from {dpath}')

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
        code_description = sample['prompt']
        golden_solution = sample['canonical_solution']
        starter_code = sample['starter_code']
        

        prompt = build_prompt(code_description, starter_code=None, openai_style=self.opt.openai_style_prompt, dialog_sep=self.opt.delimiter)

        context_vec = self.tokenizer.txt2vec(prompt)

        label = golden_solution + "\n```"
        label_vec = (self.tokenizer.txt2vec(label) + [self.tokenizer.end_token_id])

        # length_limit = self.c_trunc

        # while(len(context_vec + label_vec) > length_limit):
        #     if len(context_vec) > 0:
        #         context_vec = context_vec[1:]
        #     else:
        #         label_vec = label_vec[:-1]
        text_vec = context_vec + label_vec
        text_len = len(context_vec) + len(label_vec)
        loss_mask = [0] * len(context_vec) + [1] * len(label_vec)
        label_start_pos = len(context_vec)
        assert text_vec[label_start_pos:] == label_vec

        # if self.debug:
        
        # logging.info(f'Context Vec: {context_vec}')
        # logging.info(f'Context: {self.tokenizer.vec2txt(context_vec)}')
        # logging.info(f'Full text: {self.tokenizer.vec2txt(text_vec)}')
        # logging.info(f'Label Vec: {label_vec}')
        # logging.info(f'Label: {self.tokenizer.vec2txt(label_vec)}')
        # logging.info(f'Non-masked part: {self.tokenizer.vec2txt([tok for tok, i in zip(text_vec, loss_mask) if i > 0])}')

        output = {
            'text': sample,
            'text_encoded': self.tokenizer.vec2txt(text_vec),
            'text_vec': text_vec,
            'text_len': text_len,
            'loss_mask': loss_mask,
            'label_pos': label_start_pos,
            'label': label,
        }
        if self.no_split_dialog:
            del output['label_pos']
        return output

    def sample_generator(self):
        random.seed(None)
        need_shuffle = self.mode == 'train'
        
        # If multiprocessing dataloader is used, set data for current worker
        worker_info = get_worker_info()
        if worker_info is not None:
            self.data = self.data[worker_info.id::worker_info.num_workers]
            logging.info(f'WORKER {worker_info.id} Got {len(self.data)} samples')
            
        if need_shuffle:
            random.shuffle(self.data)
        # yield samples
        for sample in self.data:
            yield self._encode_sample(sample)
            
    def __len__(self):
        return self.size


class ChunkDataset(BaseDataset):
    # For datasets which cannot fit in the memeory
    # only a chunk (part) of data is loaded at a time
    def __init__(self, opt, accelerator, mode: str='train', **kwargs) -> None:
        super().__init__(opt, accelerator, mode, **kwargs)
        
        # example of chunk config file:
        # {
        #     "num_train": 243810681,
        #     "num_valid": 48000,
        #     "num_test": 48000,
        #     "train_chunk_ids": [1, 2, 3],
        #     "valid_chunk_ids": [4],
        #     "test_chunk_ids": [5]
        # }
        with open(os.path.join(opt.data_path, 'config.json'), 'r') as f:
            self.config: Dict[str, Any] = json.load(f)
            
        self.total_size = self.config[f'num_{mode}']
        self.chunk_ids = self.config[f'{mode}_chunk_ids']
        
    def __len__(self):
        return self.total_size

    def _load_chunk(self, chunk_i):
        dpath = os.path.join(self.opt.data_path, f'chunk_{chunk_i}.json')
        return self._load_data(dpath)

    def sample_generator(self):
        random.seed(None)
        need_shuffle = self.mode == 'train'

        # shuffle the chunks
        if need_shuffle:
            random.shuffle(self.chunk_ids)
        for chunk_i in self.chunk_ids:
            # read a chunk of data
            chunk_data: List[Tuple[str, str]] = self._load_chunk(chunk_i)

            # If distributed train, get data for current rank
            # Never shuffle the chunk_data before this
            if self.accelerator and self.accelerator.use_distributed:
                chunk_data = chunk_data[self.accelerator.process_index::self.accelerator.num_processes]
            # logging.info(f'Got {len(chunk_data)} samples from chunk_{chunk_i}')

            # If multiprocessing dataloader is used, set data for current worker
            # Never shuffle the chunk_data before this
            worker_info = get_worker_info()
            if worker_info is not None:
                chunk_data = chunk_data[worker_info.id::worker_info.num_workers]
                logging.info(f'WORKER {worker_info.id} Got {len(chunk_data)} samples from chunk_{chunk_i}')
            else:
                logging.info(f'Got {len(chunk_data)} samples from chunk_{chunk_i}')

            # now we can shuffle the data safely
            if need_shuffle:
                random.shuffle(chunk_data)
                
            # yield samples
            for sample in chunk_data:
                yield self._encode_sample(sample)
            
def get_dataloader(dataset: IterableDataset, opt):
    return DataLoader(dataset, 
                      batch_size=None, 
                      num_workers=opt.num_workers, 
                      prefetch_factor=opt.num_prefetch, 
                      pin_memory=True, )
