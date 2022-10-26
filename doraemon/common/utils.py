import torch
import random
import numpy as np
import pandas as pd


__all__ = [
    'seed_all',
    'AverageMeter',
    'StatsTracker',
]


def seed_all(seed=42, benchmark=False, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += self.val * int(n)
        self.count += int(n)
        self.avg = self.sum / self.count

    def __repr__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class StatsTracker:
    def __init__(self, prefix='', fmt=':f'):
        self.stats = {}
        self.prefix = prefix
        self.fmt = fmt

    def update(self, key, val, n=1):
        meter = self.stats.setdefault(key, AverageMeter(key, self.fmt))
        meter.update(val, n)

    def update_dict(self, dic, n=1):
        for k, v in dic.items():
            self.update(k, v, n)

    def __getattr__(self, key):
        if key in self.stats.keys():
            return self.stats[key]
        else:
            raise KeyError(f'No meter named \'{key}\'.')

    def __repr__(self):
        return ' | '.join([self.prefix + str(m) for m in self.stats.values()])
