#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import collections
import functools
import itertools
import math
import time
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from internevo.accelerator import AcceleratorType, get_accelerator
from internevo.checkpoint.utils import init_fsdp_v1
from internevo.core.context import (
    IS_REPLICA_EXPERT_DATA_PARALLEL,
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_EXPERT_DATA_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from internevo.core.context import global_context as gpc
from internevo.core.context.random import set_mode
from internevo.core.naive_amp import (
    NaiveAMPModel,
    set_fp32_attr_to_module,
    unwrap_naive_amp,
)
from internevo.core.parallel.comm.isp import (
    EmbeddingWeightParallelCommunicator,
    HeadWeightParallelCommunicator,
    ISPCommModelConfig,
    ISPCommunicator,
    ISPCommunicatorSchedulerHook,
    ISPCommunicatorWrapper,
)
from internevo.core.parallel.comm.tensor import (
    EmbeddingSequenceParallelCommunicator,
    EmbeddingTensorParallelCommunicator,
    HeadSequenceParallelCommunicator,
    HeadTensorParallelCommunicator,
    LinearRole,
    MoESequenceParallelCommunicator,
    SequenceParallelCommunicator,
    TensorParallelCommunicator,
)
from internevo.core.parallel.comm.zero import ParamAsyncBcastHandler
from internevo.core.trainer import TrainState
from internevo.data.utils import unpack_type_ids
from internevo.model.builder import create_model
from internevo.model.metrics import SchedulerMetricHook
from internevo.model.modules.embedding import Embedding1D
from internevo.model.modules.linear import (
    ColumnParallelLinear,
    GroupedColumnLinear,
    GroupedRowLinear,
    GroupedWPLinear,
    ParallelLinearWithCommExt,
    RewardModelLinear,
    RowParallelLinear,
    ScaleColumnParallelLinear,
    new_linear,
)
from internevo.model.modules.norm import new_layer_norm
from internevo.model.moe import Experts, MoE
from internevo.model.moe.moe import Qwen2MoE
from internevo.model.ops.norm import RMSNorm
from internevo.model.registry import register_model_initializer
from internevo.monitor import set_env_var
from internevo.monitor.monitor import monitor_manager as mm
from internevo.solver.optimizer import (
    FSDPadaptOptimizer,
    HybridZeroOptimizer,
    HybridZeroOptimizer_v2,
)
from internevo.solver.optimizer.compatible_adamw import new_compatible_adamw
from internevo.solver.schedulers.beta2_scheduler import Beta2Scheduler
from internevo.solver.schedulers.lr_scheduler import FineTuneCosineAnnealingWarmupLR
from internevo.train.utils import create_param_groups, map_param_block, timeout_input
from internevo.utils.common import DummyProfile, SchedulerHook, get_current_device
from internevo.utils.lazy import LazyObject
from internevo.utils.logger import get_logger
from internevo.utils.megatron_timers import megatron_timer as timer
from internevo.utils.parallel import (
    is_replica_expert_data_parallel_parameter,
    is_replica_zero_parallel_parameter,
    is_tensor_expert_data_parallel_parameter,
    is_tensor_zero_parallel_parameter,
    is_using_fsdp,
    is_using_hf,
    is_using_isp,
    is_weight_expert_data_parallel_parameter,
    is_weight_zero_parallel_parameter,
    sync_model_param,
    sync_model_replica_param_group,
)
from internevo.utils.timeout import llm_timeout
from internevo.utils.utils import TensorParallelMode

try:
    import torch_npu
except (ImportError, ModuleNotFoundError):
    pass

try:
    from torch.distributed._composable.fsdp import fully_shard

    FSDP2_SUPPORTED = True
except (ImportError, ModuleNotFoundError):
    FSDP2_SUPPORTED = False


try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    DCP_SUPPORTED = True
except (ImportError, ModuleNotFoundError):
    DCP_SUPPORTED = False


IS_INJECTED = "is_injected"

LINEAR2NEWLINEAR_NAME_MAPPING = dict(
    q_proj="wq",
    k_proj="wk",
    v_proj="wv",
    o_proj="wo",
    gate_proj="w1",
    down_proj="w2",
    up_proj="w3",
    lm_head="head",
    W_pack="wqkv",
)

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def set_param_unique_tracking_name(model):
    for chunk_id, chunk in enumerate(unwrap_naive_amp(model)):
        # Important: only works for llama-class models
        childrens = chunk.named_children()
        for _, children in childrens:
            if isinstance(children, nn.ModuleList):
                for idx, block in enumerate(children):
                    for name, child in block.named_modules():
                        if isinstance(child, (ParallelLinearWithCommExt)):
                            full_name = f"{chunk_id}.{idx}.{name}"
                            setattr(
                                child.weight,
                                "tracking_name",
                                f"{full_name}.weight",
                            )
                            if child.bias is not None:
                                setattr(
                                    child.bias,
                                    "tracking_name",
                                    f"{full_name}.bias",
                                )
            else:
                if isinstance(children, Embedding1D):
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{chunk_id}_embedding.weight",
                    )
                else:
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{chunk_id}_head.weight",
                    )


def set_fp32_attr_for_model(model: Union[nn.Module, nn.ModuleList]):
    if not isinstance(model, nn.ModuleList):
        model = [model]

    for _chunk in model:
        for _, module in _chunk.named_modules():
            if isinstance(module, (RMSNorm, nn.LayerNorm)) and gpc.config.get("use_fp32_norm", False):
                set_fp32_attr_to_module(module)


def set_parallel_attr_for_param_groups(model: Union[nn.Module, nn.ModuleList]):
    def _check_module_pure_dp(name, module):  # pylint: disable=W0613
        for param in module.parameters():
            setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

    def _check_module(name, module):
        # layer_norm
        if isinstance(module, (RMSNorm, nn.LayerNorm)):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, (MoE, Qwen2MoE)):
            for param in module.moe_layer.gate.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)
            if hasattr(module, "coefficient"):
                for param in module.coefficient.parameters():
                    setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        # embedding and head
        if isinstance(module, (Embedding1D, ScaleColumnParallelLinear)):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)

        # for moe linear module
        if isinstance(module, nn.Linear) and not isinstance(module, ParallelLinearWithCommExt):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, Experts):
            for param in module.parameters():
                if (
                    gpc.is_initialized(ParallelMode.TENSOR)
                    and not is_using_isp()
                    and getattr(gpc.config.parallel.expert, "no_tp", False)
                ):
                    setattr(param, IS_REPLICA_EXPERT_DATA_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_EXPERT_DATA_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_EXPERT_DATA_PARALLEL, True)
        # for non-moe linear module
        elif isinstance(module, ParallelLinearWithCommExt):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)

        # for vit and vit project
        if "vision_tower" in name.lower() or "vision_proj" in name.lower():
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

    for _chunk in unwrap_naive_amp(model):
        if not is_using_fsdp():
            # special case for pure dp mode
            if (
                isinstance(gpc.config.parallel["tensor"], dict)
                and gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name)
                == TensorParallelMode.mtp.name
                and gpc.get_world_size(ParallelMode.DATA) == gpc.get_world_size(ParallelMode.GLOBAL)
            ):
                _check_module_func = _check_module_pure_dp
            else:
                _check_module_func = _check_module
            # set param parallel attribute
            for name, module in _chunk.named_modules():
                _check_module_func(name, module)

            for name, param in _chunk.named_parameters():
                assert (
                    is_replica_zero_parallel_parameter(param)
                    or is_tensor_zero_parallel_parameter(param)
                    or is_weight_zero_parallel_parameter(param)
                    or is_tensor_expert_data_parallel_parameter(param)
                    or is_weight_expert_data_parallel_parameter(param)
                    or is_replica_expert_data_parallel_parameter(param)
                ), f"parameter with name: {name} has no parallel attribution."


@llm_timeout(func_name="initialize_model_and_parallel_communicator")
def initialize_model_and_parallel_communicator(
    pre_process_func: Optional[Callable] = None, post_process_func: Optional[Callable] = None
):
    """
    Initialize model with Automatic Mixed Precision.
    Returns:
        torch.nn.Module:
            The neural network model to be trained or evaluated.
        An isp communicator for managing comp/comm overlap.
    """
    if pre_process_func:
        pre_process_output = pre_process_func()

    register_model_initializer()

    model = create_model()

    if post_process_func:
        post_process_func(pre_process_output)

    return inject_model(model)


def inject_model(model):
    """
    Inject model with Automatic Mixed Precision.

    Args:
        torch.nn.Module:
            The bare neural network model to be trained or evaluated.

    Returns:
        torch.nn.Module:
            The injected neural network model to be trained or evaluated.
        An isp communicator for managing comp/comm overlap.
    """
    if hasattr(model, IS_INJECTED) and getattr(model, IS_INJECTED):
        return model

    # For non-HF cases, set tracking name for parameters
    if not is_using_hf():
        set_param_unique_tracking_name(model)

    # For non-fsdp cases, set model inject helper
    if not is_using_fsdp():
        inject_model_helper(model, inject_info=gpc.config.model.get("inject_info", None))

    # should be set before NaiveAMPModel
    set_fp32_attr_for_model(model)

    if isinstance(model, nn.ModuleList):
        model = nn.ModuleList(
            [
                NaiveAMPModel(
                    model=_m,
                    output_to_fp32=False,  # manually controlled by interleaved pipleline scheduler
                    dtype=gpc.config.model.get("dtype", torch.half),
                    sync_buffer=False,
                )
                for _m in model
            ]
        )
    else:
        model = NaiveAMPModel(
            model=model,
            output_to_fp32=gpc.is_no_pp_or_last_stage(),
            dtype=gpc.config.model.get("dtype", torch.half),
            sync_buffer=False,
        )

    set_parallel_attr_for_param_groups(model)

    # This sync is very important, cause the model weights kept in optimizer are copied
    # from the origin parameters in the memory, so we should make sure the dp sync
    # does not influence the model weights in optimizer be different with the origin parameters.
    if not is_using_fsdp() or gpc.config.parallel.fsdp.get("init_method", "cuda") == "cuda":
        sync_model_param(model)

    # This function is needed to make sure parameters that are not splitted by tensor parallelism are
    # the same across tensor parallelism.
    sync_model_replica_param_group(model)

    # Change random state mode to ParallelMode.DATA after model is built, guaranteeing the random
    # state in the same dp group are all the same.
    random_mode = ParallelMode.WEIGHT_DATA if is_using_isp() else ParallelMode.DATA
    set_mode(random_mode)

    # initialize isp communicator
    isp_communicator = initialize_parallel_communicator(model)

    model = wrap_FSDP_model(model)

    # set is_injected flag
    setattr(model, "IS_INJECTED", True)

    return model, isp_communicator


_T = TypeVar("_T")


def _submodule_filter(model: Union[nn.Module, nn.ModuleList], target_cls: Union[_T, Tuple[_T]]) -> Iterable[_T]:
    for _chunk in unwrap_naive_amp(model):
        for _module in _chunk.modules():
            if not isinstance(_module, target_cls):
                continue

            yield _module


def initialize_parallel_communicator(model: Union[nn.Module, nn.ModuleList]):
    """
    Initialize communicator for isp tensor parallel mode.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        An isp communicator for managing comp/comm overlap.
    """
    isp_communicator_wrapper = None
    _retain_out_sharded = gpc.config.model.get("parallel_output", True)

    if is_using_isp():
        isp_communicator = ISPCommunicator(
            model,
            ISPCommModelConfig(
                gpc.config.model.dtype,
                get_current_device(),
                gpc.config.model.checkpoint,
            ),
            gpc.config.parallel.weight.overlap and not is_using_fsdp(),
            gpc.get_group(ParallelMode.WEIGHT),
            is_moe=False,
            selective_ckpt_offload=gpc.config.get("selective_checkpoint_offload", False),
            early_reduce_scatter_release=gpc.config.parallel.weight.early_reduce_scatter_release,
        )
        # register communicator for isp column parallel linear.
        ColumnParallelLinear.register_cls_communicator(isp_communicator)
        # row parallel linear will not be used.
        RowParallelLinear.register_cls_communicator(None)
        _head_communicator = HeadWeightParallelCommunicator(
            weight_process_group=gpc.get_group(ParallelMode.WEIGHT),
            seq_process_group=gpc.get_group(ParallelMode.TENSOR),
            retain_out_sharded=_retain_out_sharded,
        )
        _embedding_communicator = EmbeddingWeightParallelCommunicator(ParallelMode.WEIGHT)

        if gpc.config.model.get("num_experts", 1) > 1:
            # register communicator for moe isp column parallel linear.
            # NOTE: this wil overwrite registed communicator
            moe_isp_communicator = ISPCommunicator(
                model,
                ISPCommModelConfig(
                    gpc.config.model.dtype,
                    get_current_device(),
                    gpc.config.model.checkpoint,
                ),
                gpc.config.parallel.expert_weight.overlap,
                gpc.get_group(ParallelMode.EXPERT_WEIGHT),
                is_moe=True,
                early_reduce_scatter_release=gpc.config.parallel.expert_weight.early_reduce_scatter_release,
            )
            for moe in _submodule_filter(model, Experts):
                for column_linear in _submodule_filter(moe, (ColumnParallelLinear, GroupedWPLinear)):
                    column_linear.register_communicator(moe_isp_communicator)
                for row_linear in _submodule_filter(moe, RowParallelLinear):
                    row_linear.register_communicator(None)

            isp_communicator_wrapper = ISPCommunicatorWrapper([isp_communicator, moe_isp_communicator])
        else:
            isp_communicator_wrapper = ISPCommunicatorWrapper([isp_communicator])

    # register communictor for mtp/msp/fsp linear.

    # tensor parallel
    if gpc.config.parallel.tensor.mode == TensorParallelMode.mtp.name:
        ColumnParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN)
        )
        RowParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW)
        )

        if gpc.config.model.get("num_experts", 1) > 1:
            GroupedColumnLinear.register_cls_communicator(
                TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN)
            )
            GroupedRowLinear.register_cls_communicator(
                TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW)
            )
            GroupedWPLinear.register_cls_communicator(None)
            # treat as sequence paralle if no_tp
            if gpc.config.parallel.expert.no_tp:
                _column_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.COLUMN
                )
                _row_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.ROW
                )
                for moe in _submodule_filter(model, MoE):
                    # 1. the linear in MoE degrades as no tp communication pattern
                    for column_linear in _submodule_filter(moe, ColumnParallelLinear):
                        column_linear.register_communicator(_column_communicator)
                    for row_linear in _submodule_filter(moe, RowParallelLinear):
                        row_linear.register_communicator(_row_communicator)
                    # 2. register MoESequenceParallelCommunicator for MoE layer
                    MoESequenceParallelCommunicator(ParallelMode.TENSOR, reverse=True).register_module_hook(moe)

        _head_communicator = HeadTensorParallelCommunicator(ParallelMode.TENSOR, _retain_out_sharded)
        _embedding_communicator = EmbeddingTensorParallelCommunicator(ParallelMode.TENSOR)
    # sequence parallel
    if gpc.config.parallel.tensor.mode in (TensorParallelMode.msp.name, TensorParallelMode.fsp.name):
        save_total_input_as_activation = gpc.config.parallel.tensor.mode == TensorParallelMode.msp.name

        ColumnParallelLinear.register_cls_communicator(
            SequenceParallelCommunicator(
                process_group=gpc.get_group(ParallelMode.TENSOR),
                role=LinearRole.COLUMN,
                save_total_input_as_activation=save_total_input_as_activation,
            )
        )
        RowParallelLinear.register_cls_communicator(
            SequenceParallelCommunicator(
                gpc.get_group(ParallelMode.TENSOR),
                role=LinearRole.ROW,
                save_total_input_as_activation=save_total_input_as_activation,
            )
        )
        if gpc.config.model.get("num_experts", 1) > 1:
            GroupedColumnLinear.register_cls_communicator(
                SequenceParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    role=LinearRole.COLUMN,
                    save_total_input_as_activation=save_total_input_as_activation,
                )
            )
            GroupedRowLinear.register_cls_communicator(
                SequenceParallelCommunicator(
                    gpc.get_group(ParallelMode.TENSOR),
                    role=LinearRole.ROW,
                    save_total_input_as_activation=save_total_input_as_activation,
                )
            )
            GroupedWPLinear.register_cls_communicator(None)
            if gpc.config.parallel.expert.no_tp:
                _column_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.COLUMN
                )
                _row_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.ROW
                )
                for moe in _submodule_filter(model, MoE):
                    # 1. the linear in MoE degrades as no tp communication pattern
                    for column_linear in _submodule_filter(moe, ColumnParallelLinear):
                        column_linear.register_communicator(_column_communicator)
                    for row_linear in _submodule_filter(moe, RowParallelLinear):
                        row_linear.register_communicator(_row_communicator)

        _head_communicator = HeadSequenceParallelCommunicator(
            ParallelMode.TENSOR, _retain_out_sharded, save_total_input_as_activation
        )

        _embedding_communicator = EmbeddingSequenceParallelCommunicator(ParallelMode.TENSOR)

    # register communitorc for embedding layer.
    if not is_using_fsdp():
        for embedding in _submodule_filter(model, Embedding1D):
            _embedding_communicator.register_module_hook(embedding)

    # register communictor for head layer.
    ScaleColumnParallelLinear.register_cls_communicator(_head_communicator)
    RewardModelLinear.register_cls_communicator(_head_communicator)

    return isp_communicator_wrapper


@llm_timeout(func_name="initialize_optimizer")
def initialize_optimizer(model: Union[nn.Module, nn.ModuleList], isp_communicator: ISPCommunicatorWrapper = None):
    """
    Initialize optimizer.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        A tuple of (optimizer, beta2_scheduler, lr_scheduler).
    """

    adam_cfg = gpc.config.adam
    zero_cfg = gpc.config.hybrid_zero_optimizer
    grad_scal_cfg = gpc.config.grad_scaler
    use_apex_adam = getattr(gpc.config, "use_apex_adam", False)

    if "use_split_tensor_optim" in zero_cfg and zero_cfg.use_split_tensor_optim:
        map_param_block(model)

    params = create_param_groups(model, adam_cfg.weight_decay)

    naive_optimizer = new_compatible_adamw(
        params=params,
        lr=adam_cfg.lr,
        betas=(adam_cfg.adam_beta1, adam_cfg.adam_beta2),
        eps=adam_cfg.adam_eps,
        use_apex_adam=use_apex_adam,
    )

    if (
        zero_cfg.overlap_sync_grad
        and gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
        and gpc.is_pipeline_first_stage() is False
    ):
        # When pipeline parallelism is enabled, we prefer to only enable optimizer
        # gradient communication overlap in the first stage, to avoid amplifying
        # the communication overhead stage by stage in cases where the optimizer
        # communication overhead is greater than the compute overhead.
        # For pipeline stages except the first, even if overlap is not enabled,
        # their gradient synchronization overhead can be well hidden by
        # the inherent bubbles of pipeline parallelism.
        zero_cfg.overlap_sync_grad = False

    if zero_cfg.overlap_sync_param:
        param_bcast_sync_handler = ParamAsyncBcastHandler(ParallelMode.ZERO1, model, isp_communicator)
    else:
        param_bcast_sync_handler = None

    if not is_using_fsdp():
        if (
            "use_split_tensor_optim" not in gpc.config.hybrid_zero_optimizer
            or not gpc.config.hybrid_zero_optimizer.use_split_tensor_optim
        ):
            optimizer = HybridZeroOptimizer(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
        else:
            optimizer = HybridZeroOptimizer_v2(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
    else:
        optimizer = FSDPadaptOptimizer(
            naive_optimizer,
            grad_scal_cfg=grad_scal_cfg,
            zero_cfg=zero_cfg,
        )

    beta2_scheduler = Beta2Scheduler(optimizer=naive_optimizer, **gpc.config.beta2_scheduler)

    lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, **gpc.config.lr_scheduler)

    return optimizer, beta2_scheduler, lr_scheduler


def get_scheduler_hooks(metric, zero_optim, isp_communicator_wrapper) -> List[SchedulerHook]:
    scheduler_hooks: List[SchedulerHook] = []

    if metric is not None:
        scheduler_hooks.append(
            SchedulerMetricHook(
                metric=metric,
                skip=(
                    gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
                    and hasattr(gpc.config.model, "num_chunks")
                    and gpc.config.model.num_chunks > 1
                    and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
                ),
            ),
        )

    if isp_communicator_wrapper is not None:
        for isp_communicator in isp_communicator_wrapper.isp_communicators:
            if isp_communicator is not None and isp_communicator.overlap:
                scheduler_hooks.append(ISPCommunicatorSchedulerHook(isp_communicator, zero_optim))

    return scheduler_hooks


@llm_timeout(func_name="load_new_batch")
def load_new_batch(train_dl: DataLoader, train_iter: Iterable, train_state: TrainState):
    """
    Load and return the new batch data based on training data loader.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader for training.
        train_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
        train_state (TrainState): Current training state.

    Returns: A batch data and the updated train_iter.
    """

    timer("batch-gen").start()
    try:
        batch = next(train_iter)  # structure is ({'input_ids': Tensor, 'cu_seqlens': Tensor}, Tensor)
        if hasattr(train_state, "batch_sampler_iter"):
            next(train_state.batch_sampler_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)
        train_state.num_consumed_samples_in_epoch = 0
        if hasattr(train_state, "batch_sampler"):
            train_state.batch_sampler.batch_count = 0
            train_state.batch_sampler.num_consumed_samples_in_epoch = 0
            train_state.batch_sampler_iter = iter(train_state.batch_sampler)
            next(train_state.batch_sampler_iter)
    timer("batch-gen").stop()

    if batch[0].get("type_ids", None) is not None:
        # if use_packed_dataset is False, we need to unpack type_ids
        if not gpc.config.data.use_packed_dataset:
            batch[0]["type_ids"] = unpack_type_ids(batch[0]["type_ids"], batch[0]["cu_seqlens"])

    return batch, train_iter


def initialize_llm_profile(profiling: bool = False, start_time: str = None):
    """Initialize and return the profiler context manager instance."""

    if profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        schedule_config = {"wait": 1, "warmup": 1, "active": 1, "repeat": 1, "skip_first": 3}
        trace_path = (
            f"RUN/{gpc.config.JOB_NAME}/{start_time}/traces/rank{gpc.get_global_rank()}_"
            f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            f"wp{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
            f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}"
        )
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            llm_profile = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(**schedule_config),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
            logger.info(f"Do profiling for NPU on rank {gpc.get_global_rank()}!")
        else:
            llm_profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(**schedule_config),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
                with_stack=True,
                with_modules=True,
                profile_memory=True,
            )
            logger.info(f"Do profiling for GPU on rank {gpc.get_global_rank()}!")
    else:
        llm_profile = DummyProfile()

    return llm_profile


@llm_timeout(func_name="record_current_batch_training_metrics")
def record_current_batch_training_metrics(
    get_tflops_func,
    logger,
    writer,
    success_update,
    batch_count,
    batch,
    train_state,
    optimizer,
    beta2_scheduler,
    engine,
    start_time,
    very_begining_time,
    loss,
    moe_loss,
    grad_norm,
    metric,
):
    """
    Print some training metrics of current batch.
    """

    set_env_var(key="LAST_ACTIVE_TIMESTAMP", value=int(time.time()))

    timer.store_last_timers()
    if success_update in (0, True):
        train_state.num_consumed_tokens += batch[1].nelement() * gpc.get_world_size(ParallelMode.DATA)
    if gpc.is_no_pp_or_last_stage():
        acc_perplex = metric.get_metric()

    if success_update and gpc.is_rank_for_log():
        lr = optimizer.param_groups[0]["lr"]
        if hasattr(engine.optimizer, "grad_scaler"):
            scaler = engine.optimizer.grad_scaler._scale.item()
        elif hasattr(engine.optimizer.optim, "grad_scaler"):
            scaler = engine.optimizer.optim.grad_scaler._scale.item()

        num_tokens_in_batch = batch[1].nelement()
        real_num_tokens = math.ceil(acc_perplex.pop("real_token_num") / gpc.get_world_size(ParallelMode.GLOBAL))
        num_samples_in_batch = sum([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        max_length_in_batch = max([(b[1:] - b[:-1]).max().item() for b in batch[0]["cu_seqlens"]])
        max_samples_in_batch = max([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        min_samples_in_batch = min([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        time_cost = time.time() - start_time
        tk_per_gpu = round(
            num_tokens_in_batch * gpc.get_world_size(ParallelMode.DATA) / gpc.get_world_size(ParallelMode.GLOBAL),
            4,
        )
        tgs_statistic = train_state.tgs_statistic
        tgs_statistic["sum_step"] += 1
        tgs_statistic["sum_tg"] += tk_per_gpu
        tgs_statistic["total_time"] = time.time() - very_begining_time
        tgs_statistic["sum_last_tg_10"] += tk_per_gpu
        tgs_statistic["sum_last_time_10"] += time_cost
        tgs_statistic["sum_last_tg_50"] += tk_per_gpu
        tgs_statistic["sum_last_time_50"] += time_cost
        tgs_statistic["SMA_tg_50"] += tk_per_gpu
        tgs_statistic["SMA_time_50"] += time_cost
        tgs_statistic["SMA_tg_50_list"].append(tk_per_gpu)
        tgs_statistic["SMA_time_50_list"].append(time_cost)
        if tgs_statistic["sum_step"] > 50:
            tgs_statistic["SMA_tg_50"] -= tgs_statistic["SMA_tg_50_list"][0]
            tgs_statistic["SMA_time_50"] -= tgs_statistic["SMA_time_50_list"][0]
            tgs_statistic["SMA_tg_50_list"].popleft()
            tgs_statistic["SMA_time_50_list"].popleft()

        last_tgs_1 = round(tk_per_gpu / time_cost, 2)
        tgs_statistic["sum_tgs"] += last_tgs_1

        if tgs_statistic["sum_step"] % 10 == 0:
            tgs_statistic["last_tgs_10"] = round(tgs_statistic["sum_last_tg_10"] / tgs_statistic["sum_last_time_10"], 2)
            tgs_statistic["sum_last_tg_10"] = 0
            tgs_statistic["sum_last_time_10"] = 0

        if tgs_statistic["sum_step"] % 50 == 0:
            tgs_statistic["last_tgs_50"] = round(tgs_statistic["sum_last_tg_50"] / tgs_statistic["sum_last_time_50"], 2)
            tgs_statistic["sum_last_tg_50"] = 0
            tgs_statistic["sum_last_time_50"] = 0

        last_tgs_10 = tgs_statistic["last_tgs_10"]
        last_tgs_50 = tgs_statistic["last_tgs_50"]

        tgs_all = round(tgs_statistic["sum_tg"] / tgs_statistic["total_time"], 2)
        tgs_avg = round(tgs_statistic["sum_tgs"] / tgs_statistic["sum_step"], 2)
        tgs_SMA = round(tgs_statistic["SMA_tg_50"] / tgs_statistic["SMA_time_50"], 2)

        tflops = get_tflops_func(time_cost)

        tgs_origin = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / time_cost,
            2,
        )

        real_tgs = round(
            real_num_tokens / time_cost,
            2,
        )

        infos = {
            "tflops": tflops,
            "step": batch_count,
            "loss": loss.item() - moe_loss.item() if moe_loss is not None else loss.item(),
            "real_tgs": real_tgs,
            "tgs (tokens/gpu/second)": tgs_origin,
            "tgs/last_tgs_1": last_tgs_1,
            "tgs/tgs_all": tgs_all,
            "tgs/tgs_avg": tgs_avg,
            "tgs/tgs_SMA": tgs_SMA,
            "tgs/last_tgs_10": last_tgs_10,
            "tgs/last_tgs_50": last_tgs_50,
            "lr": lr,
            "loss_scale": scaler,
            "grad_norm": grad_norm,
        }
        if moe_loss is not None:
            infos["moe_loss"] = moe_loss.item()

        infos["micro_num"] = len(batch[1])
        infos["num_consumed_tokens"] = train_state.num_consumed_tokens
        infos["inf_nan_skip_batches"] = train_state.inf_nan_skip_batches
        infos["num_samples_in_batch"] = num_samples_in_batch  # the number of batches which have the most samples
        infos["largest_length"] = max_length_in_batch  # the longest input
        infos["largest_batch"] = max_samples_in_batch  # the batch with the most samples
        infos["smallest_batch"] = min_samples_in_batch
        infos["adam_beta2"] = beta2_scheduler.get_beta2()

        fwd_bwd_time = round(timer("fwd-bwd").elapsed(), 2)
        infos["fwd_bwd_time"] = fwd_bwd_time
        bwd_time = round(timer("bwd").elapsed(), 2)
        infos["bwd_time"] = bwd_time

        for key, value in acc_perplex.items():
            infos[key] = value

        line = ""
        for key, value in infos.items():
            line += f"{key}={value} "
            if isinstance(value, dict):
                writer.add_scalars(key=key, value=value, step=train_state.step_count)
            else:
                writer.add_scalar(key=key, value=value, step=train_state.step_count)

        logger.info(line)

        # if loss spike occurs, send alert info to feishu
        mm.monitor_loss_spike(
            alert_address=gpc.config.monitor.alert.feishu_alert_address,
            step_count=batch_count,
            cur_step_loss=loss.item(),
        )


def inject_embed(model: nn.Module, inject=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding) and not isinstance(child, Embedding1D):
                msg = (
                    f"To get parallel training enabled, module {name} of type {nn.Embedding.__name__} "
                    + f"is required to be replaced with {Embedding1D.__name__}."
                )
                if inject:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = Embedding1D(
                            num_embeddings=child.num_embeddings,
                            embedding_dim=child.embedding_dim,
                            padding_idx=child.padding_idx,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def inject_linear(model: nn.Module, inject=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and not isinstance(child, ParallelLinearWithCommExt):
                msg = (
                    f"To get parallel training enabled, module {name} of type {nn.Linear.__name__} "
                    + f"is required to be replaced with {new_linear.__name__}."
                )
                if inject:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = new_linear(
                            name=LINEAR2NEWLINEAR_NAME_MAPPING.get(name, name),
                            in_features=child.in_features,
                            out_features=child.out_features,
                            bias=child.bias is not None,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def inject_norm(model: nn.Module, inject=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if "RMSNorm" in cls_name:
                msg = (
                    f"To re-use unified RMSNorm implementation, {cls_name} "
                    + f"is suggested to be replaced with {new_layer_norm.__name__}."
                )
                if inject:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = new_layer_norm(
                            norm_type="rmsnorm",
                            normalized_shape=child.weight.shape,
                            eps=child.variance_epsilon,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def inject_config(model: nn.Module) -> None:
    # Compatibility for Vision-Language Model
    if hasattr(model.config, "text_config"):
        llm_cfg = model.config.text_config
    else:
        llm_cfg = model.config
    gpc.config.model.vocab_size = gpc.config.VOCAB_SIZE = llm_cfg.vocab_size
    gpc.config.model.hidden_size = gpc.config.HIDDEN_SIZE = llm_cfg.hidden_size
    gpc.config.model.num_layers = gpc.config.NUM_LAYER = llm_cfg.num_hidden_layers
    # Compatibility for Mamba
    if hasattr(llm_cfg, "num_attention_heads"):
        gpc.config.model.num_attention_heads = gpc.config.NUM_ATTENTION_HEAD = llm_cfg.num_attention_heads
    gpc.config.model.mlp_ratio = gpc.config.MLP_RATIO = llm_cfg.intermediate_size / llm_cfg.hidden_size
    # For models that use GQA
    if hasattr(llm_cfg, "num_key_value_heads"):
        gpc.config.model.num_kv_attention_heads = gpc.config.NUM_KV_ATTENTION_HEAD = llm_cfg.num_key_value_heads


def _get_modules_to_materialize(
    root_module: nn.Module,
    ignored_modules: Set[nn.Module],
) -> List[nn.Module]:
    # Run BFS to collect the modules to materialize via `reset_parameters()`,
    # stopping at any module with FSDP already applied or at ignored modules.
    modules_to_materialize: List[nn.Module] = []
    queue = collections.deque([root_module])
    visited_modules: Set[nn.Module] = {root_module}
    while queue:
        module = queue.popleft()
        modules_to_materialize.append(module)
        for child_module in module.children():
            if child_module not in visited_modules and child_module not in ignored_modules:
                visited_modules.add(child_module)
                queue.append(child_module)
    return modules_to_materialize


def _materialize_meta_module(
    root_module: nn.Module,
    ignored_modules: Set[nn.Module],
    device_id: Optional[torch.device],
) -> None:
    # Run default meta device initialization
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    module = None
    try:
        # Assume that each module's `reset_parameters()` only initializes its
        # own parameters and not those of its children
        with torch.no_grad():
            for module in modules_to_materialize:
                # As a contract to the user, only call `reset_parameters()` if
                # the module has directly managed parameters/buffers
                module_state_iter = itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))
                has_module_states = len(list(module_state_iter)) > 0
                if has_module_states:
                    module.to_empty(device=device_id, recurse=False)
                    module.reset_parameters()  # type: ignore[operator]
    except BaseException as e:
        logger.warning(
            "Unable to call `reset_parameters()` for module on meta "
            f"device with error {str(e)}. Please ensure that your module of"
            f"type {type(module)} implements a `reset_parameters()` method."  # type: ignore[possibly-undefined]
        )
        raise e


def wrap_FSDP_model(model: Union[nn.Module, nn.ModuleList]):
    if is_using_fsdp():
        assert isinstance(model, nn.Module), "Currently FSDP does not support pipeline parallel."
        wrap_cls = tuple(
            LazyObject(warp_cls["mod"], warp_cls["mod_cls"]).build() for warp_cls in gpc.config.get("fsdp_wrap_cls", [])
        )
        fsdp_mode = gpc.config.parallel.fsdp.get("mode", "v1")
        fsdp_init_method = gpc.config.parallel.fsdp.get("init_method", "cuda")

        if fsdp_mode == "v1":
            model = FSDP(
                module=model,
                process_group=gpc.get_group(ParallelMode.GLOBAL),
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO2: SHARD_GRAD_OP, ZeRO3: FULL_SHARD
                auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=set(wrap_cls)),
                sync_module_states=fsdp_init_method != "cuda",  # sync model paramters
                forward_prefetch=True,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                limit_all_gathers=True,
                use_orig_params=True,
                device_id=None if fsdp_init_method == "cuda" else get_current_device(),  # needed for sync_module_states
            )
            # For FSDP v1, to get ckpt resuming work normally, we do dummy forward.
            # This hack is needed due to FSDP v1 lazy initialization in model construction.
            # FYI: https://github.com/pytorch/pytorch/issues/113496
            model = init_fsdp_v1(model, get_current_device())
        elif FSDP2_SUPPORTED and fsdp_mode == "v2":
            fsdp_kwargs = {
                "reshard_after_forward": True,  # ZeRO2: False, ZeRO3: True
            }
            for module in model.modules():
                if isinstance(module, wrap_cls):
                    fully_shard(module, **fsdp_kwargs)
            fully_shard(model, **fsdp_kwargs)
            if fsdp_init_method == "meta":
                _materialize_meta_module(model, set(), get_current_device())
            elif fsdp_init_method == "cpu":
                model.to(get_current_device())
        else:
            raise ValueError(f"Unsupported FSDP mode: {fsdp_mode}")

        if is_using_hf() and not gpc.config.ckpt.get("auto_resume", False):
            load_ckpt_info = gpc.config.ckpt.load_ckpt_info
            load_ckpt_path = load_ckpt_info.get("path", None)
            load_ckpt_content = load_ckpt_info.get("content", [])
            if load_ckpt_path:
                assert load_ckpt_content == (
                    "model",
                ), "If auto_resume=False and checkpoint path is given, only model can be loaded"
                if DCP_SUPPORTED:
                    hf = gpc.config.hf
                    mod = LazyObject(hf.mod, hf.mod_cls)
                    mod = mod.build()
                    state_dict = mod.from_pretrained(
                        pretrained_model_name_or_path=load_ckpt_path, use_safetensors=True
                    ).state_dict()
                    state_dict = {f"model.{key}": state_dict[key].clone().detach() for key in state_dict}
                    set_model_state_dict(
                        model=model, model_state_dict=state_dict, options=StateDictOptions(full_state_dict=True)
                    )
                    del state_dict
                    internlm_accelerator.empty_cache()
                else:
                    raise RuntimeError("DCP is not supported in this version of PyTorch.")

    return model


def inject_model_helper(model: Union[nn.Module, nn.ModuleList], inject_info: Optional[Dict] = None) -> None:
    """
    Inject model helper functions.

    Args:
        model (Union[nn.Module, nn.ModuleList]):
            For built-in models, it is nn.Module for no pp and nn.ModuleList for pp.
            For injected models, it is nn.Module.
        inject_info (Optional[Dict]): configurations for injected_models.
    """
    # parse inject_info
    if inject_info is not None:
        inject = inject_info.get("inject", False)
        interactive = inject_info.get("interactive", False)
        modules = inject_info.get("modules", [])
        reset_params = inject_info.get("reset_params", False)
        extra_linear2newlinear = inject_info.get("extra_linear2newlinear", {})
    else:
        inject = False
        interactive = False
        modules = []
        reset_params = False
        extra_linear2newlinear = {}

    LINEAR2NEWLINEAR_NAME_MAPPING.update(extra_linear2newlinear)

    inject_funcs = {
        "embed": inject_embed,
        "linear": inject_linear,
        "norm": inject_norm,
    }

    # inject config
    if inject:
        inject_config(model)

    if not isinstance(model, nn.ModuleList):
        model = [model]
    for _chunk in model:
        # Special case for pure dp mode: skip
        if (
            isinstance(gpc.config.parallel["tensor"], dict)
            and gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.mtp.name
            and gpc.get_world_size(ParallelMode.DATA) == gpc.get_world_size(ParallelMode.GLOBAL)
        ):
            continue
        # In-place replacement or check for modules: "embed", "linear", "norm"
        # (1) If inject=True, in-place replacement
        # (2) If inject=False, check
        for mod in modules:
            inject_funcs[mod](_chunk, inject, interactive)
        # reset parameters if needed, model should have reset_parameters() method
        if reset_params:
            _chunk.reset_parameters()
    for _chunk in model:
        # If model is initialized on cpu, model should be moved to cuda device after injection
        if not next(_chunk.parameters()).is_cuda:
            _chunk.to(get_current_device())

    # print injected model
    if inject and gpc.is_rank_for_log():
        logger.info(
            f"inject is enabled, please check the model carefully, "
            f"if there are any problems, please report issue to us. "
            f"The injected model is \n {model}"
        )
