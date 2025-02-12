# Copyright (c) InternLM. All rights reserved.
from internevo.core.context import global_context as gpc
from internevo.core.trainer import TrainState
from internevo.utils.utils import DataType


def get_train_state(dataloader):
    # initialize and resume train state
    if gpc.config.data.type in [
        DataType.tokenized.name,
        DataType.streaming.name,
        DataType.megatron.name,
        DataType.mocked.name,
    ]:
        train_state = TrainState(gpc.config, dataloader.batch_sampler)
    else:
        raise ValueError(f"dataset type {gpc.config.data.type} is not supported")

    return train_state
