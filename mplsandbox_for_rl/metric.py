from typing import List, Optional, Any, Tuple, Union, Dict
from typing import Counter as TCounter
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.metrics import BLEU
from nltk.translate import bleu_score as nltkbleu
import rouge
import math
from collections import Counter, deque
from functools import reduce
import logging
# import tensorflow_io
from accelerate import Accelerator
import torch
from torch.utils.tensorboard import SummaryWriter

class Metric:
    def __init__(self) -> None:
        pass

    def add(self, val):
        raise NotImplementedError

    def val(self) -> float:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def compute(self, val: Any):
        return val

    def __add__(self, other):
        raise NotImplementedError
    
    def __radd__(self, other):
        # other + self
        return self.__add__(other)

class ExponentialMovingAverageMetric(Metric):
    def __init__(self, factor, val=None, denom=1):
        self.factor = factor
        self.val_ = val
        # this denominator is different from MeanMetric's and 
        # only accumulate for calculating the mean EMA score between different nodes/processes
        self.denominator: int = denom
        
    def add(self, val):
        if self.val_ is None:
            self.val_ = self.compute(val)
        else:
            self.val_ = self.val_ * self.factor + self.compute(val) * (1. - self.factor)
    
    def val(self) -> float:
        if self.val_ is None:
            return None
        return self.val_ / self.denominator
    
    def sync(self, val):
        self.val_ = val
        self.denominator = 1
    
    def reset(self):
        self.val_ = None
        
    def __add__(self, other: 'ExponentialMovingAverageMetric'):
        assert self.factor == other.factor, 'It seems that you are aggregating no-related EMA metrics since their factors are not equal'
        return ExponentialMovingAverageMetric(self.factor, self.val_ + other.val_, self.denominator + other.denominator)
    
class SpanSumMetric(Metric):
    def __init__(self, size, buffer=None, sum_=0, denom=1) -> None:
        super().__init__()
        self.size = size
        self.buffer = deque() if buffer is None else deque(buffer)
        self.sum = sum_
        self.denominator = denom
        
    def add(self, val):
        val = self.compute(val)
        self.buffer.append(val)
        self.sum += val
        while len(self.buffer) > self.size:
            self.sum -= self.buffer.popleft()
    
    def val(self) -> float:
        return self.sum / self.denominator
    
    def reset(self):
        self.sum = 0
        self.buffer.clear()
        self.denominator = 1
        
    def __add__(self, other: 'SpanSumMetric'):
        assert self.size == other.size, 'It seems that you are aggregating no-related span metrics objs since their span size are not equal'
        return SpanSumMetric(self.size, None, self.sum + other.sum, self.denominator + other.denominator)
    
class MeanMetric(Metric):
    def __init__(self, num=0, denom=0):
        self.numerator = num
        self.denominator: int = denom

    def add(self, val: Any):
        self.numerator += self.compute(val)
        self.denominator += 1

    def many(self, vals: List[Any], denoms: Optional[List[int]] = None):
        if denoms is None:
            denoms = [1] * len(vals)
        assert len(vals) == len(denoms)

        for v, n in zip(vals, denoms):
            self.numerator += self.compute(v)
            self.denominator += n
    
    def val(self):
        if self.denominator == 0:
            return 0
        return self.numerator / self.denominator

    def reset(self):
        self.numerator = self.denominator = 0

    def __add__(self, other: 'MeanMetric'):
        # self + other
        return MeanMetric(self.numerator + other.numerator, self.denominator + other.denominator)


class SumMetric(Metric):
    def __init__(self, sum_=0):
        self.sum_ = sum_

    def add(self, val):
        self.sum_ += self.compute(val)

    def many(self, vals: List[Any]):
        self.sum_ += sum(self.compute(v) for v in vals)

    def val(self):
        return self.sum_

    def reset(self):
        self.sum_ = 0

    def __add__(self, other: 'SumMetric'):
        return SumMetric(self.sum_ + other.sum_)


class RealtimeMetric(Metric):
    def __init__(self, val=0):
        self.v = val

    def add(self, val):
        self.v = self.compute(val)
        
    def many(self, vals: List[Any]):
        self.add(vals[-1])
    
    def val(self):
        return self.v

    def reset(self):
        self.v = 0

    def __add__(self, other):
        return RealtimeMetric(self.v)
    
class CIDErDMetric(Metric):
    def __init__(self, lang, ngram=4, sigma=15, guesses=None, answers=None) -> None:
        super().__init__()
        self.lang = lang
        self.ngram = ngram
        self.sigma = sigma
        self.guesses = guesses or []
        self.answers = answers or []
    
    def add(self, val: Tuple[str, List[str]]):
        guess, answers = val
        assert isinstance(guess, str) and isinstance(answers, list)
        
        guess = [normalize_answer(guess, lang=self.lang)]
        answers = [normalize_answer(a, lang=self.lang) for a in answers]
        self.guesses.append(guess)
        self.answers.append(answers)
        
    def reset(self):
        self.guesses.clear()
        self.answers.clear()
    
    def val(self) -> float:
        from metric_utils import CiderD
        assert len(self.guesses) == len(self.answers)
        cider = CiderD(df='corpus', sigma=self.sigma)
        score, _ = cider.compute_score(res=[{'image_id': i, 'caption': s} for i, s in zip(range(len(self.guesses)), self.guesses)],
                                       gts={i: s for i, s in zip(range(len(self.answers)), self.answers)})
        return score
    
    def __add__(self, other: 'CIDErDMetric'):
        return CIDErDMetric(self.lang, ngram=self.ngram, sigma=self.sigma, 
                            guesses=self.guesses + other.guesses, answers=self.answers + other.answers)
        


class BleuMetric(MeanMetric):
    language_map = {'cn': 'zh', 'en': '13a', 'intl': 'intl'}
    def __init__(self, b=4, backend='sacre', lang='zh'):
        super().__init__()
        b = int(b)
        self.backend = backend
        self.lang = self.language_map.get(lang, lang)

        if backend == 'sacre':
            self.bleu = BLEU(lowercase=True, tokenize=self.lang, max_ngram_order=b, effective_order=True)
        elif backend == 'nltk':
            self.weights = [1 / b for _ in range(b)]
        else:
            raise ValueError

    def compute(self, val: Tuple[str, List[str]]):
        guess, answers = val
        answers = [a.strip() for a in answers if a]
        if not answers or not guess:
            return 0.

        if self.backend == 'nltk':
            score = nltkbleu.sentence_bleu(
                [normalize_answer(a, lang=self.lang).split(' ') for a in answers],
                normalize_answer(guess, lang=self.lang).split(' '),
                smoothing_function=nltkbleu.SmoothingFunction().method7,
                weights=self.weights,
            )
        elif self.backend == 'sacre':
            score = self.bleu.sentence_score(hypothesis=guess, references=answers).score
        else:
            raise ValueError

        return score
    
class CorpusBleuMetric(Metric):
    def __init__(self, b=4, backend='sacre', lang='zh', guesses=None, answers=None):
        super().__init__()
        b = int(b)
        self.b = b
        self.backend = backend
        self.lang = BleuMetric.language_map.get(lang, lang)

        if backend == 'sacre':
            self.bleu = BLEU(lowercase=True, tokenize=self.lang, max_ngram_order=b, effective_order=True)
        elif backend == 'nltk':
            self.weights = [1 / b for _ in range(b)]
        else:
            raise ValueError
        
        self.guesses = [] if guesses is None else guesses
        self.answers = [] if answers is None else answers
    
    def reset(self):
        self.guesses.clear()
        self.answers.clear()
        
    def add(self, val: Tuple[str, List[str]]):
        guess, answers = val
        self.guesses.append(guess)
        self.answers.append(answers)
        
    def val(self) -> float:
        if self.backend == 'nltk':
            score = nltkbleu.corpus_bleu([[normalize_answer(a, lang=self.lang).split(' ') for a in answer] for answer in self.answers],
                                         [normalize_answer(guess, lang=self.lang).split(' ') for guess in self.guesses],
                                         smoothing_function=nltkbleu.SmoothingFunction().method7, weights=self.weights,)
        elif self.backend == 'sacre':
            score = self.bleu.corpus_score(self.guesses, list(zip(*self.answers))).score
        else:
            raise ValueError
        return score
    
    def __add__(self, other: 'CorpusBleuMetric'):
        return CorpusBleuMetric(self.b, backend=self.backend, lang=self.lang, 
                                guesses=self.guesses + other.guesses, answers=self.answers + other.answers)


class RougeMetric(MeanMetric):
    def __init__(self, r, lang='zh'):
        assert r in ('rouge-1', 'rouge-2', 'rouge-l')
        super().__init__()
        self.rouge_type = r
        self.rouge = rouge.Rouge(metrics=[r])
        self.lang = lang

    def compute(self, val: Tuple[str, List[str]]):
        guess, answers = val
        guess = normalize_answer(guess, self.lang).strip()
        answers = (normalize_answer(a, self.lang).strip() for a in answers)
        answers = [a for a in answers if a]
        if not answers or not guess or not all(answers):
            return 0.
        try:
            score = max(self.rouge.get_scores(guess, a)[0][self.rouge_type]['r'] for a in answers)
        except ValueError:
            logging.warn(f'Error sample {(guess, answers)}')
            return 0.
        return score


class F1Metric(MeanMetric):
    def __init__(self, lang='zh'):
        super().__init__()
        self.lang = lang

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def compute(self, val: Tuple[str, List[str]]):
        guess, answers = val
        g_tokens = normalize_answer(guess, self.lang).split()
        a_tokens = [normalize_answer(a, lang=self.lang).split() for a in answers]
        if not g_tokens or not a_tokens:
            return 0.

        scores = [F1Metric._prec_recall_f1_score(g_tokens, a) for a in a_tokens]
        return max(f1 for p, r, f1 in scores)
        

class InterDistinctMetric(Metric):
    def __init__(self, d, lang='zh', counts: Optional[TCounter] = None):
        super().__init__()
        self.lang = lang
        self.distinct_type = d
        self.counts: TCounter[Tuple] = Counter() if counts is None else counts

    @staticmethod
    def _ngram(seq: List[str], n):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])

    def add(self, val: Any):
        self.counts += self.compute(val)
        
    def many(self, vals: List[Any]):
        for v in vals:
            self.add(v)

    def compute(self, val: str):
        tokens = normalize_answer(val, lang=self.lang).split()
        return Counter(InterDistinctMetric._ngram(tokens, self.distinct_type))

    def val(self):
        return max(len(self.counts), 1e-10) / max(sum(self.counts.values()), 1e-5)

    def __add__(self, other: 'InterDistinctMetric'):
        return InterDistinctMetric(self.distinct_type, self.lang, counts=self.counts + other.counts)

    def reset(self):
        self.counts = Counter()
    

class PPLMetric(MeanMetric):
    def val(self):
        # super().val is nll loss
        try:
            return math.exp(super().val())
        except OverflowError:
            return super().val()

    def __add__(self, other):
        return PPLMetric(self.numerator + other.numerator, self.denominator + other.denominator)

class Metrics():
    # TRAIN_METRICS = ['loss', 'ppl', 'token_acc', 'lr', 'gnorm', 'loss_scale', 'clen', 'llen', 'ctrunc', 'ltrunc', 'tpb', 'ctpb', 'ltpb', 'expb', 'ups', 'total_exs']
    # GENERATION_METRIC = ['f1', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'rouge-1', 'rouge-2', 'rouge-l', 'dist-1', 'dist-2']
    # EVAL_METRICS_SKIP_GENERATION = ['loss', 'ppl', 'token_acc', 'clen', 'llen', 'ctrunc', 'ltrunc', 'total_exs']
    TRAIN_METRICS = ['loss', 'ppl', 'token_acc', 'lr', 'clen', 'ctrunc', 'tpb', 'expb', 'ups', 'total_exs']
    GENERATION_METRIC = ['f1', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'rouge-1', 'rouge-2', 'rouge-l', 'dist-1', 'dist-2', 'cider']
    EVAL_METRICS_SKIP_GENERATION = ['loss', 'ppl', 'token_acc', 'clen', 'ctrunc', 'total_exs']
    EVAL_METRICS = EVAL_METRICS_SKIP_GENERATION + GENERATION_METRIC
    tb_writer = None # tensorboard writer

    def __init__(self, opt: Dict[str, Any], accelerator: Accelerator, mode='train'):
        self.metrics: Dict[str, Metric] = {}
        self.mode = mode
        self.opt = opt
        self.accelerator = accelerator

        # for normal train
        if mode == 'train':
            self.metrics.update({n: self._get_metric_obj(n) for n in self.TRAIN_METRICS})
        # for eval with no generation
        elif mode == 'eval_skip_generation':
            self.metrics.update({n: self._get_metric_obj(n) for n in self.EVAL_METRICS_SKIP_GENERATION})
        # for eval
        elif mode == 'eval':
            self.metrics.update({n: self._get_metric_obj(n) for n in self.EVAL_METRICS})
        else:
            raise ValueError

        if Metrics.tb_writer is None and opt.tensorboard_logdir is not None and self.accelerator.is_main_process:
            Metrics.tb_writer = SummaryWriter(opt.tensorboard_logdir)

    def add_additional_metric(self, metric_name: str, metric_obj: Metric):
        assert metric_name not in self.metrics
        self.metrics[metric_name] = metric_obj
        
    def remove_useless_metric(self, metric_name: str):
        assert metric_name in self.metrics
        del self.metrics[metric_name]

    def record_metric(self, metric_name: str, val: Any):
        self.metrics[metric_name].add(val)

    def record_metric_many(self, metric_name: str, vals: List[Any], counts: Optional[List[int]] = None):
        assert hasattr(self.metrics[metric_name], 'many'), f'Cannot call "many" for Metric: {metric_name}'

        if counts is None:
            self.metrics[metric_name].many(vals)
        else:
            self.metrics[metric_name].many(vals, counts)
            
    def get_metric(self, metric_name: str):
        return self.metrics[metric_name].val()

    def reset(self, no_reset: List[str] = ['total_exs']):
        for k, v in self.metrics.items():
            if k not in no_reset:
                v.reset()
                
    def all_gather_metrics(self) -> Dict[str, float]:
        with torch.no_grad():
            metrics_tensor = {k: torch.tensor([v.val()], device=self.accelerator.device) for k, v in self.metrics.items()}
            
            if self.accelerator.use_distributed:
                gathered_metrics = self.accelerator.gather(metrics_tensor)
                for metric_name, gathered_tensor in gathered_metrics.items():
                    if metric_name in ('loss', 'ppl', 'token_acc', 'gnorm', 'clen', 'llen', 'ctrunc', 'ltrunc', 'ups', \
                                       'f1', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'cider', 'rouge-1', 'rouge-2', 'rouge-l', 'dist-1', 'dist-2', 'lr'):
                        gathered_metrics[metric_name] = gathered_tensor.float().mean()
                    elif metric_name in ('total_exs', 'tpb', 'ctpb', 'ltpb', 'expb'):
                        gathered_metrics[metric_name] = gathered_tensor.sum()
                    else:
                        gathered_metrics[metric_name] = gathered_tensor.float().mean()
            else:
                gathered_metrics = metrics_tensor
                                        
            gathered_metrics = {k: v.item() for k, v in gathered_metrics.items()}
            # print(gathered_metrics)
        return gathered_metrics
    
    def write_tensorboard(self, global_step: int, gathered_metrics: Dict[str, float] = None):
        results = self.all_gather_metrics() if gathered_metrics is None else gathered_metrics
        if self.tb_writer is not None:
            for k, scalar in results.items():
                title = f"{k}/{'train' if 'train' == self.mode else 'eval'}"
                self.tb_writer.add_scalar(tag=title, scalar_value=scalar, global_step=global_step)
                
    def flush(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def display(self, global_step: int, data_size: Optional[int] = None, gathered_metrics: Dict[str, float] = None):
        if not self.accelerator.is_main_process:
            return
        results = self.all_gather_metrics() if gathered_metrics is None else gathered_metrics
        log_str = ''
        if data_size is not None and 'total_exs' in results:
            print(f"---------- Step: {global_step}, Epoch: {(results['total_exs'] / data_size):.2f} ----------")
        else:
            print(f'---------- Step: {global_step} ----------')
        for k, value in results.items():
            if isinstance(value, float):
                if k == 'lr':
                    value = f'{value:.3e}'
                else:
                    value = f'{value:.4f}'
            log_str += f'{k}: {value}\t'
        print(log_str)        
        return results        

    def _get_metric_obj(self, metric_name: str):
        if metric_name in ('loss', 'token_acc', 'gnorm', 'clen', 'llen', 'ctrunc', 'ltrunc', 'tpb', 'ctpb', 'ltpb', 'expb', 'ups'):
            return MeanMetric()
        if metric_name in ('total_exs',):
            return SumMetric()
        if metric_name == 'ppl':
            return PPLMetric()
        if metric_name in ('lr', 'loss_scale'):
            return RealtimeMetric()
        if 'bleu' in metric_name:
            type_ = int(metric_name[-1])
            if self.opt.bleu_level == 'sentence':
                return BleuMetric(type_, backend=self.opt.bleu_backend, lang=self.opt.lang)
            elif self.opt.bleu_level == 'corpus':
                return CorpusBleuMetric(type_, backend=self.opt.bleu_backend, lang=self.opt.lang)
            else:
                raise ValueError
        if 'rouge' in metric_name:
            return RougeMetric(metric_name, lang=self.opt.lang)
        if 'dist' in metric_name:
            type_ = int(metric_name[-1])
            return InterDistinctMetric(type_, lang=self.opt.lang)
        if 'f1' == metric_name:
            return F1Metric(lang=self.opt.lang)
        if 'cider' == metric_name:
            return CIDErDMetric(lang=self.opt.lang, sigma=self.opt.cider_sigma)
        raise ValueError
        

def normalize_answer(s, lang: str) -> str:
    """
    tokenize text
    """

    s = s.lower()
    if lang == 'zh':
        s = TokenizerZh()(s)
    elif lang == 'intl':
        s = TokenizerV14International()(s)
    elif lang in ('en', '13a'):
        s = Tokenizer13a()(s)
    else:
        raise ValueError
    return s

if __name__ == '__main__':
    def add(a, b):
        return a + b

    opt = {
        'tensorboard_logdir': None, 'bleu_backend': 'sacre', 'lang': 'zh'
    }
    # test gather metrics
    test_1 = [MeanMetric(1, 1), MeanMetric(1, 2), MeanMetric(1, 3)]
    test_2 = [SumMetric(1), SumMetric(2), SumMetric(3)]
    test_3 = [RealtimeMetric(1), RealtimeMetric(2), RealtimeMetric(3)]
    test_4 = [PPLMetric(1, 1), PPLMetric(2, 1), PPLMetric(3, 1)]
    print(reduce(add, test_1).val())
    print(reduce(add, test_2).val())
    print(reduce(add, test_3).val())
    print(reduce(add, test_4).val())

    # test generation metrics
    bleu_metric = BleuMetric(b=4)
    rouge_metric = RougeMetric('rouge-1')
    dist_metric = InterDistinctMetric(1)
    f1_metric = F1Metric()

    case = ('我不是大笨蛋蛋', ['我是大笨蛋'])
    bleu_metric.add(case)
    rouge_metric.add(case)
    dist_metric.add(case[0])
    f1_metric.add(case)
    print(bleu_metric.val(), rouge_metric.val(), dist_metric.val(), f1_metric.val())

    # test all
    metrics = Metrics(opt)
    for _ in range(2):
        for i, m_name in enumerate(['loss', 'ppl', 'token_acc', 'lr', 'gnorm', 'clen', 'llen', 'tpb', 'ctpb', 'ltpb', 'expb', 'ups', 'total_exs']):
            metrics.record_metric(m_name, i)
    
    metrics.display(100)

    metrics.reset()
    metrics.record_metric_many('loss', [1, 2, 3], [3, 2, 1])
    metrics.record_metric_many('ppl', [1, 2, 3], [3, 2, 1])
    metrics.display(100, 1000)
