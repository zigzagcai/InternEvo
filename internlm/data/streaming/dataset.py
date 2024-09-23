import itertools
import sys

import datasets
import numpy as np
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from transformers import AutoTokenizer


class StreamingDataset(Dataset):
    """
    Streaming and on-the-fly tokenize dataset
    """

    def __init__(
        self,
        dataset_path,
        tokenizer_path,
        model_max_length,
        content_name="text",
        subset_name=None,
        split="train",
        buffer_size=1000,
    ):
        self.dataset = datasets.load_dataset(dataset_path, data_dir=subset_name, split=split, streaming=True)
        self.dataset = split_dataset_by_node(
            self.dataset, rank=gpc.get_local_rank(ParallelMode.DATA), world_size=gpc.get_world_size(ParallelMode.DATA)
        )
        self.content_name = content_name
        self.buffer_size = buffer_size
        self.senior_iterator = iter(self)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tokenizer.model_max_length = model_max_length

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                yield from self._tokenize(buffer)
                buffer = []

        if buffer:
            yield from self._tokenize(buffer)

    def __len__(self):
        return sys.maxsize

    def _tokenize(self, samples):
        texts = [sample[self.content_name] for sample in samples]
        tokenized_outputs = self.tokenizer(texts, truncation=True)
        for i in range(len(samples)):
            if len(tokenized_outputs["input_ids"][i]) > 0:
                yield {key: tokenized_outputs[key][i] for key in tokenized_outputs}

    def __getitem__(self, _):
        return next(self.senior_iterator)


class StreamingDatasetPackSampleWithPad(Dataset):
    """
    Streaming dataset with pack_sample_into_one=False
    StreamingDatasetPackSampleWithPad streaming and on-the-fly consumes data samples, then aggregates 
    samples of different lengths together based on the packed_length=seq_len*micro_bsz using pad mode.
    seq_len = 5
    micro_bsz = 2
    packed_length = 5 * 2 = 10
    
    Original dataset:
    [1, 2]
    [3, 4]
    [5, 6, 7]
    [8, 9, 10, 11, 12]
    [13, 14]
    --->
    
    Packed dataset:
    input_ids=[1, 2, 3, 4, 5, 6, 7, 0, 0, 0], cu_seqlens=[0, 2, 4, 7, 10]
    input_ids=[8, 9, 10, 11, 12, 13, 14, 0, 0, 0], cu_seqlens=[0, 5, 7, 10]
    """

    def __init__(self, dataset, seq_len, micro_bsz, pad_token_id=0):
        self.dataset = dataset
        self.seq_len = seq_len
        self.micro_bsz = micro_bsz
        self.pad_token_id = pad_token_id
        self.senior_iterator = iter(self)

    def __iter__(self):
        input_ids = []
        cu_seqlens = [0]
        labels = []
        for sample in self.dataset:
            if len(input_ids + sample["input_ids"]) > self.micro_bsz * self.seq_len:
                input_ids = input_ids + [self.pad_token_id] * (self.micro_bsz * self.seq_len - len(input_ids))
                cu_seqlens = (
                    cu_seqlens + [self.micro_bsz * self.seq_len]
                    if cu_seqlens[-1] < self.micro_bsz * self.seq_len
                    else cu_seqlens
                )
                labels = labels + [-100] * (self.micro_bsz * self.seq_len - len(labels))
                yield {
                    "input_ids": input_ids,
                    "cu_seqlens": cu_seqlens,
                    "indexes": list(
                        itertools.chain(*[np.arange(l2 - l1) for l1, l2 in zip(cu_seqlens[:-1], cu_seqlens[1:])])
                    ),
                    "labels": labels,
                    "type_ids": [0] * (self.micro_bsz * self.seq_len),
                }
                input_ids = sample["input_ids"]
                cu_seqlens = [0, len(sample["input_ids"])]
                labels = [w if w > 0 else -100 for w in sample["input_ids"]][1:] + [-100]
            else:
                input_ids = input_ids + sample["input_ids"]
                cu_seqlens.append(len(sample["input_ids"]) + cu_seqlens[-1])
                labels = labels + [w if w > 0 else -100 for w in sample["input_ids"]][1:] + [-100]

        if len(input_ids) > 0:
            input_ids = input_ids + [self.pad_token_id] * (self.micro_bsz * self.seq_len - len(input_ids))
            cu_seqlens = (
                cu_seqlens + [self.micro_bsz * self.seq_len]
                if cu_seqlens[-1] < self.micro_bsz * self.seq_len
                else cu_seqlens
            )
            labels = labels + [-100] * (self.micro_bsz * self.seq_len - len(labels))
            yield {
                "input_ids": input_ids,
                "cu_seqlens": cu_seqlens,
                "indexes": list(
                    itertools.chain(*[np.arange(l2 - l1) for l1, l2 in zip(cu_seqlens[:-1], cu_seqlens[1:])])
                ),
                "labels": labels,
                "type_ids": [0] * (self.micro_bsz * self.seq_len),
            }

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, _):
        return next(self.senior_iterator)


class StreamingDatasetPackSampleIntoOneWithCut(Dataset):

    """
    Streaming dataset with pack_sample_into_one=True
    
    StreamingDatasetPackSampleIntoOneWithCut streaming and on-the-fly consumes data samples, then aggregates 
    samples of different lengths together based on the packed_length=seq_len*micro_bsz using cut mode.
    seq_len = 5
    micro_bsz = 2
    packed_length = 5 * 2 = 10
    
    Original dataset:
    [1, 2]
    [3, 4]
    [5, 6, 7]
    [8, 9, 10, 11, 12]
    [13, 14]
    --->
    
    Packed dataset:
    input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cu_seqlens=[0, 5, 10]
    input_ids=[11, 12, 13, 14, 0, 0, 0, 0, 0, 0], cu_seqlens=[0, 5, 10]
    
    """

    def __init__(self, dataset, seq_len, micro_bsz, pad_token_id=0):
        self.dataset = dataset
        self.seq_len = seq_len
        self.micro_bsz = micro_bsz
        self.pad_token_id = pad_token_id
        self.senior_iterator = iter(self)

    def __iter__(self):
        input_ids = []
        labels = []
        for sample in self.dataset:
            if len(input_ids + sample["input_ids"]) > self.micro_bsz * self.seq_len:
                cut_input_ids = sample["input_ids"][:self.micro_bsz * self.seq_len - len(input_ids)]
                if len(cut_input_ids) > 0:
                    input_ids = input_ids + cut_input_ids
                    labels = labels + [w if w > 0 else -100 for w in cut_input_ids][1:] + [-100]
                yield {
                    "input_ids": input_ids,
                    "cu_seqlens": [i * self.seq_len for i in range(self.micro_bsz + 1)],
                    "indexes": list(range(self.seq_len)) * self.micro_bsz,
                    "labels": labels,
                    "type_ids": [0] * (self.micro_bsz * self.seq_len),
                }
                cut_residual_input_ids = sample["input_ids"][self.micro_bsz * self.seq_len - len(input_ids):]
                if len(cut_residual_input_ids) > 0:
                    input_ids = cut_residual_input_ids
                    labels = [w if w > 0 else -100 for w in cut_residual_input_ids][1:] + [-100]
                else:
                    input_ids = []
                    labels = []
            else:
                input_ids = input_ids + sample["input_ids"]
                labels = labels + [w if w > 0 else -100 for w in sample["input_ids"]][1:] + [-100]

        if len(input_ids) > 0:
            input_ids = input_ids + [self.pad_token_id] * (self.micro_bsz * self.seq_len - len(input_ids))
            labels = labels + [-100] * (self.micro_bsz * self.seq_len - len(labels))
            assert len(labels) == self.micro_bsz * self.seq_len
            yield {
                "input_ids": input_ids,
                "cu_seqlens": [i * self.seq_len for i in range(self.micro_bsz + 1)],
                "indexes": list(range(self.seq_len)) * self.micro_bsz,
                "labels": labels,
                "type_ids": [0] * (self.micro_bsz * self.seq_len),
            }

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, _):
        return next(self.senior_iterator)
