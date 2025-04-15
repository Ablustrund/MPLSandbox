from transformers import AutoTokenizer
from typing import List
from utils import write_log_info_on_rank0

class HFPretrainedTokenizer():
    def __init__(self, opt):
        self.opt = opt
        write_log_info_on_rank0(f"Loading vocab from huggingface {opt.hf_model_name}", log_once=True)
        
        from transformers.models.llama.tokenization_llama import LlamaTokenizer
        # self.hf_tokenizer = LlamaTokenizer.from_pretrained(
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
                opt.hf_model_name, 
                trust_remote_code=True,
                # bos_token='</s>',
                # eos_token='</s>',
                # unk_token='</s>',
                # pad_token='[PAD]'
            )
        # self.hf_tokenizer.pad_token = self.hf_tokenizer.unk_token
        # self.hf_tokenizer.pad_token_id = self.hf_tokenizer.unk_token_id

        # Add special tokens
        self.override_special_tokens()
        self.vocab_size = len(self.hf_tokenizer.get_vocab())
        write_log_info_on_rank0(f'Special tokens: {self.hf_tokenizer.special_tokens_map},\nwhere {list(zip(self.hf_tokenizer.all_special_tokens, self.hf_tokenizer.all_special_ids))}', log_once=True)
        
    @property
    def gmask_token(self):
        return self.hf_tokenizer.gmask_token
    
    @property
    def gmask_token_id(self):
        return self.hf_tokenizer.gmask_token_id
    
    @property
    def null_token(self):
        return self.hf_tokenizer.pad_token
    
    @property
    def end_token(self):
        return self.hf_tokenizer.eos_token
    
    @property
    def unk_token(self):
        return self.hf_tokenizer.unk_token
    
    @property
    def start_token(self):
        return self.hf_tokenizer.bos_token
    
    @property
    def null_token_id(self):
        return self.hf_tokenizer.pad_token_id
    
    @property
    def end_token_id(self):
        return self.hf_tokenizer.eos_token_id
    
    @property
    def unk_token_id(self):
        return self.hf_tokenizer.unk_token_id
    
    @property
    def start_token_id(self):
        return self.hf_tokenizer.bos_token_id
    
    
    def override_special_tokens(self):
        pass
        
    def txt2vec(self, text: str):
        # return [token for token in self.hf_tokenizer.encode(text, add_special_tokens=False) if token != self.unk_token_id]
        # text = self.detokenizer.detokenize(text.split())
        return self.hf_tokenizer.encode(text, add_special_tokens=False)

    def vec2txt(self, vector: List[int], skip_special=False):
        text = self.hf_tokenizer.decode(vector, skip_special_tokens=skip_special)
        return text
    

# if __name__ == '__main__':
#     class OPT:
#         def __init__(self, model_path, model_type):
#             self.hf_model_name = model_path
#             self.model_type = model_type

#     opt = OPT('decapoda-research/llama-13b-hf', 'llama')
#     tokenizer = HFPretrainedTokenizer(opt)
#     print(tokenizer.vocab_size)
#     print(tokenizer.start_token, '->', tokenizer.start_token_id)
#     print(tokenizer.end_token, '->', tokenizer.end_token_id)
#     print(tokenizer.unk_token, '->', tokenizer.unk_token_id)
#     print(tokenizer.null_token, '->', tokenizer.null_token_id)
