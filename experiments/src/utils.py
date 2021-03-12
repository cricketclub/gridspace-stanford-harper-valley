import os
import json
import torch
import shutil
import functools
import numpy as np
from collections import Counter, OrderedDict

memoized = functools.lru_cache(maxsize=None)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def copy_checkpoint(folder='./', filename='checkpoint.pth.tar',
                    copyname='copy.pth.tar'):
    shutil.copyfile(os.path.join(folder, filename),
                    os.path.join(folder, copyname))


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def edit_distance(src_seq, tgt_seq):
    src_len, tgt_len = len(src_seq), len(tgt_seq)
    if src_len == 0: return tgt_len
    if tgt_len == 0: return src_len

    dist = np.zeros((src_len+1, tgt_len+1))
    for i in range(1, tgt_len+1):
        dist[0, i] = dist[0, i-1] + 1
    for i in range(1, src_len+1):
        dist[i, 0] = dist[i-1, 0] + 1
    for i in range(1, src_len+1):
        for j in range(1, tgt_len+1):
            cost = 0 if src_seq[i-1] == tgt_seq[j-1] else 1
            dist[i, j] = min(
                dist[i,j-1]+1,
                dist[i-1,j]+1,
                dist[i-1,j-1]+cost,
            )
    return dist


def get_cer(hypotheses, hypothesis_lengths, references, reference_lengths):
    assert len(hypotheses) == len(references)
    cer = []
    for i in range(len(hypotheses)):
        if len(hypotheses[i]) > 0:
            dist_i = edit_distance(
                hypotheses[i][:hypothesis_lengths[i]],
                references[i][:reference_lengths[i]],
            )
            # CER divides the edit distance by the length of the true sequence
            cer.append((dist_i[-1, -1] / float(reference_lengths[i])))
        else:
            cer.append(1)  # since we predicted empty 
    return np.mean(cer)


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


def frozen_params(module):
    for p in module.parameters():
        p.requires_grad = False


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
