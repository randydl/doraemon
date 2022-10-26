import torch
import numpy as np
import pandas as pd


__all__ = [
    'BalancedSampler',
    'DatasetFromSampler',
]


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, inputs, sampling_mode='same', get_labels=None):
        labels = inputs if get_labels is None else get_labels(inputs)
        counts = np.unique(labels, return_counts=True)[-1]

        if sampling_mode == 'down':
            length = len(counts) * counts.min()
        elif sampling_mode == 'over':
            length = len(counts) * counts.max()
        elif sampling_mode == 'same':
            length = len(labels)
        elif isinstance(sampling_mode, int) and sampling_mode > 0:
            length = sampling_mode
        else:
            raise ValueError("sampling_mode must be one of ['down', 'over', 'same'] or a positive integer")

        self.length = length
        self.weights = 1 / counts[labels] / len(counts)

    def __iter__(self):
        indexs = range(len(self.weights))
        sample = np.random.choice(indexs, self.length, replace=True, p=self.weights)
        return iter(sample.tolist())

    def __len__(self):
        return self.length


class DatasetFromSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __getitem__(self, index):
        indexs = list(self.sampler)
        return self.dataset[indexs[index]]

    def __len__(self):
        return len(self.sampler)
