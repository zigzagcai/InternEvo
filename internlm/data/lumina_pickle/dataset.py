import copy
import json
import os
import pickle
import traceback
import yaml
import h5py
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class LuminaPickleDataset(Dataset):
    def __init__(self, data_yaml: str, micro_batch_size: int, seq_len: int, base_path: str = None, cache_on_disk=True):
        logger.info(f"read data meta yaml from {data_yaml}, {base_path=}")
        if base_path is not None:
            self.base_path = base_path
        self.data_yaml = data_yaml
        self.meta_list, self.record_list = self._load_data_yaml(data_yaml)
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len

        self.cache_on_disk = cache_on_disk
        self.rank = gpc.get_local_rank(ParallelMode.DATA)

        if self.cache_on_disk:
            cache_dir = self._get_cache_dir(data_yaml)
            if self.rank == 0:
                self._collect_data_and_save_to_cache(cache_dir)
            torch.distributed.barrier()
            self.meta_list, self.record_list = self._load_data_from_cache(cache_dir)
        else:
            cache_dir = None
            self.meta_list, self.record_list = self._load_data_yaml(data_yaml)

    def __len__(self) -> int:
        total_len = sum(meta["len"] for meta in self.meta_list)
        logger.info(f"debug {total_len=}")
        for meta in self.meta_list:
            logger.info(f"{meta['len']=}")

        return total_len

    def __getitem__(self, idx: int):
        meta_idx, idx_in_meta = self.tie_index_to_meta(idx)

        try:
            return self.get_item_func(meta_idx, idx_in_meta)
        except Exception as e:
            logger.info(
                f"Item {idx} errored, record:\n"
                f"{self.record_list[meta_idx][idx_in_meta]}\n"
                f"Error:\n"
                f"{traceback.format_exc()}"
            )
            if idx_in_meta != 0:
                return self[idx - 1]
            else:
                return self[idx + self.meta_list[meta_idx]["len"] - 1]

    def _load_data_yaml(self, data_yaml: str):
        meta_list = []
        record_list = []
        with open(data_yaml, "r") as yaml_fin:
            data_meta = yaml.load(yaml_fin, Loader=yaml.FullLoader)
            for meta in data_meta["META"]:
                record_json_path = meta["path"]
                if not os.path.exists(record_json_path) and hasattr(self, "base_path") and self.base_path is not None:
                    record_json_path = os.path.join(self.base_path, record_json_path)
                with open(record_json_path) as record_json_fin:
                    record_json = json.load(record_json_fin)
                    record_list.append(record_json)
                    meta["len"] = len(record_json)

                if "type" not in meta:
                    meta["type"] = "default"
                meta["item_len_list"] = [r["len"] for r in record_json]
                meta_list.append(meta)
        return meta_list, record_list

    def tie_index_to_meta(self, idx: int):
        # Initialize the starting index
        start_idx = 0

        # Iterate through the list of dictionaries
        for i, meta in enumerate(self.meta_list):
            # Calculate the ending index for the current collection
            end_idx = start_idx + meta["len"]

            # Check if the given index falls within the current collection
            if start_idx <= idx < end_idx:
                # Calculate the new index within the current collection
                new_index = idx - start_idx
                return i, new_index

            # Update the starting index for the next collection
            start_idx = end_idx

        # If the index is out of range of all collections, raise an error
        raise IndexError("Index out of range")

    def get_item_func(self, meta_idx, idx_in_meta):
        record_item = self.record_list[meta_idx][idx_in_meta]

        if self.cache_on_disk:
            record_item = json.loads(record_item)
            file_path = record_item["file"]
        else:
            file_path = record_item["file"]

        if not os.path.exists(file_path) and hasattr(self, "base_path") and self.base_path is not None:
            file_path = os.path.join(self.base_path, file_path)
        with open(file_path, "rb") as f:
            data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return {
            "tokens": self.extend_data_to_packed_length(tokens, self.seq_len, 0.0),
            "cu_seqlens": [i * self.seq_len for i in range(2)],
            "indexes": list(range(self.seq_len)),
            "labels": self.extend_data_to_packed_length(labels, self.seq_len, -100.0),
            "type_ids": self.extend_data_to_packed_length([0], self.seq_len, 0.0)
        }

    def informal_data_format(self, data: list):
        return self.extend_data_to_packed_length(data, self.seq_len)

    def extend_data_to_packed_length(self, data: list, packed_length, padding):
        if len(data) > packed_length:
            return data[:packed_length]
        if len(data) == packed_length:
            return data
        # len(data) < packed_length
        return data + [padding] * (packed_length - len(data))

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def _get_cache_dir(config_path):
        config_identifier = config_path
        disallowed_chars = ["/", "\\", ".", "?", "!"]
        for _ in disallowed_chars:
            config_identifier = config_identifier.replace(_, "-")
        cache_dir = f"./xllmx_data_cache/{config_identifier}"
        return cache_dir

    def _collect_data_and_save_to_cache(self, cache_dir):
        if (Path(cache_dir) / "data.h5").exists() and (Path(cache_dir) / "ready").exists():
            # off-the-shelf annotation cache exists
            logger.warning(
                f"Use existing h5 data cache: {Path(cache_dir)}\n"
                f"Note: if the actual data defined by the data config has changed since your last run, "
                f"please delete the cache manually and re-run this experiment, or the data actually used "
                f"will not be updated"
            )
            return

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        meta_list, record_list = self._load_data_yaml(self.data_yaml)

        # when cache on disk, rank0 saves items to an h5 file
        logger.info(f"start to build data cache to: {Path(cache_dir)}")
        with h5py.File(Path(cache_dir) / "data.h5", "w") as file:
            dt = h5py.vlen_dtype(str)
            for i, annotations in enumerate(record_list):
                serialized_ann = [json.dumps(_) for _ in annotations]
                h5_ann = file.create_dataset(f"ann{i}", (len(serialized_ann),), dtype=dt)
                h5_ann[:] = serialized_ann

            file.create_dataset("meta_collection", data=json.dumps(meta_list))
        with open(Path(cache_dir) / "ready", "w") as f:
            f.write("ready")
        logger.info(f"data cache built")

    @staticmethod
    def _load_data_from_cache(cache_dir):
        while not (Path(cache_dir) / "ready").exists():
            # cache has not yet been completed by rank 0
            assert torch.distributed.get_rank() != 0
            time.sleep(1)
        cache_file = h5py.File(Path(cache_dir) / "data.h5", "r")
        meta_collection = json.loads(cache_file["meta_collection"].asstr()[()])
        records_collection = [cache_file[f"ann{i}"] for i in range(len(meta_collection))]

        return meta_collection, records_collection

    @staticmethod
    def predict_item_token_length(data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()