import re
import numpy as np
from bisect import bisect_left
import torch
import rouge
import itertools

aggregator = 'Best'
apply_avg = aggregator == 'Avg'
apply_best = aggregator == 'Best'
evaluator = rouge.Rouge(metrics=['rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=apply_avg,
                        apply_best=apply_best,
                        alpha=0.5,  # Default F1_score
                        weight_factor=1.2,
                        stemming=True) # remember we shall edit the LREC paper to get an adopted rouge-w 

def rouge_w(pred, truth):
    # rouge
    scores = evaluator.get_scores(' '.join(map(str, pred)), ' '.join(map(str, truth)))
    return scores['rouge-w']['f']
    
def acc(pred, truth):
    return np.mean(np.array(pred) == np.array(truth))
    
def kendall_tau(pred, truth):
    s_t = set([i for i in itertools.combinations(truth, 2)])
    s_p = set([i for i in itertools.combinations(pred, 2)])
    cn_2 = len(pred) * (len(pred) - 1) / 2
    pairs = len(s_p) - len(s_p.intersection(s_t))
    tau = 1 - 2 * pairs / cn_2
    return tau
    
def pmr(pred, truth):
    return pred == truth

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def bundle_part_to_batch(all_bundle, l = None, r = None):
    """Convert all_bundle[l:r] to a batch of inputs.
    
    Args:
        all_bundle (list of Bundles): Data in ``Bundle'' format.
        l (int, optional): Left endpoint of the interval. Defaults to None.
        r (int, optional): Right endpoint of the interval. Defaults to None.
    
    Returns:
        tuple: A batch of inputs.
    """
    # When we want to put in the whole batch
    if l is None:
        l, r = 0, len(all_bundle.passage_length)
    num_samples = r - l

    batch_length = sum([x for x in all_bundle.pairs_num[l:r]])
    max_length = max(all_bundle.max_sample_len[l:r])
    input_ids = torch.zeros((batch_length, max_length), dtype = torch.long)
    token_type_ids = torch.zeros((batch_length, max_length), dtype = torch.long)
    masked_ids = torch.zeros((batch_length, max_length), dtype = torch.long)
    
    ground_truth = all_bundle.ground_truth[l:r]
    passage_length = all_bundle.passage_length[l:r]
    pairs_num = all_bundle.pairs_num[l:r]

    l_pairs = sum(all_bundle.pairs_num[:l])
    r_pairs = sum(all_bundle.pairs_num[:r])
    pairs_list = all_bundle.pairs_list[l_pairs: r_pairs]

    sep_positions = all_bundle.sep_positions[l_pairs:r_pairs]
    
    for i in range(l_pairs,r_pairs):
        length = len(all_bundle.input_ids[i])
        input_ids[i - l_pairs, :length] = torch.tensor(all_bundle.input_ids[i], dtype = torch.long)
        token_type_ids[i - l_pairs, :length] = torch.tensor(all_bundle.token_type_ids[i], dtype = torch.long)
        masked_ids[i - l_pairs, :length] = 1
    return input_ids, token_type_ids, masked_ids, pairs_list, sep_positions, ground_truth, passage_length, pairs_num

class WindowMean:
    def __init__(self, window_size = 50):
        self.array = []
        self.sum = 0
        self.window_size = window_size
    def update(self, x):
        self.array.append(x)
        self.sum += x
        if len(self.array) > self.window_size:
            self.sum -= self.array.pop(0)
        return self.sum / len(self.array)