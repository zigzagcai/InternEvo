import collections
import itertools
import math
from collections.abc import Sequence
from contextlib import nullcontext
from functools import partial
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from einops import rearrange, repeat 
from omegaconf import DictConfig, ListConfig

from src.args import InitArgs, ModelArgs, TrainerArgs
from src.data.iterator.utils import CondUsageType


from src.model.dit_transformer import (
    ContextEmbedder,
    PositionalEmbeddingsBlock,
    TimestepEmbedder,
)

from src.model.dit_transformer_blocks import DiTTransformerBlock, t2i_modulate

from src.model.patch_embedders.build import build_patch_embedder

from src.model.transformer import (
    _Identity,
    build_norm_fn,
    get_init_fn,
    log_accumulation,
    Transformer, 
)

from src.model.utils import dit_select_cp_seq_chunk

from src.parallelism import parallel_state as mpu

from src.parallelism.tensor_parallel import (
    ColumnParallelLinear, 
    copy_to_tensor_model_parallel_region, 
    gather_from_sequence_parallel_region, 
    linear_with_grad_accumulation_and_async_allreduce, 
    scatter_to_sequence_parallel_region, 
)

from src.utils import norm_

from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

logger = getLogger()

try:
    from apex.normalization.fused_layer_norm import (
        fused_rms_norm, 
        fused_rms_norm_affine, 
        FusedRMSNorm,
        manual_rms_norm,
    )

    class ParalleLRMSNorm(FusedRMSNorm):
        def __init__(self, *args, sequence_parallel: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self.sequence_parallel = sequence_parallel
        def forward(self, input_):
            weight = self.weight
            if self.sequence_parallel and weight is not None:
                weight = copy_to_tensor_model_parallel_region(weight)

            if not input_.is_cuda:
                return manual_rms_norm(input_, self.normalized_shape, weight, self.eps)
            
            if self.elementwise_affine:
                return fused_rms_norm_affine(
                    input_, weight, self.normalized_shape, self.eps
                )
            else:
                return fused_rms_norm(input_, self.normalized_shape, self.eps)
            
except (ImportError, ModuleNotFoundError):
    ParallelRMSNorm = None
    logger.warn("apex not found")

try:
    from xformers.ops import MoeColumnParallelLinear, MoeRowParallelLinear
except (ImportError, ModuleNotFoundError):
    MoeColumnParallelLinear = None
    MoeRowParallelLinear = None
    print("moe_matmul not found")

try:
    from xformers import ops as xops 
    from xformers.ops import AttentionBias 
except (ImportError, ModuleNotFoundError):
    AttentionBias = Any
    logger.warn("xFormers AttentionBias op not found")

class LinGen(Transformer):
    def __init__(
         self, 
         args: ModelArgs,   
    ):
        args.output_size = (
            args.patch_size**2 * args.output_channels * args.temporal_patch_size
        )

        self.factorize_positional_embeddings = args.factorize_positional_embeddings
        self.sincos_positional_embeddings = args.sincos_positional_embeddings
        self.remove_positional_embeddings = args.remove_positional_embeddings
        self.factorized_attn = args.get("factorized_attn", False)

        self.use_final_norm = args.get("use_final_norm", True)

        max_height = args.in_height // args.patch_size
        max_width = args.in_width // args.patch_size
        max_latent_frames = args.num_latent_frames // args.temporal_patch_size
        args.max_length = max_height * max_width * max_latent_frames
        
        self.dit_sequence_parallel = args.dit_sequence_parallel
        self.dit_context_parallel_text = args.dit_context_parallel_text 
        if self.factorized_attn:
            from src.model.factorized_attention import (
                FactorizedDiTTransformerBlock, 
                FactorizedPositionalEmbeddingsBlock,
            )
        
            super().__init__(args, FactorizedDiTTransformerBlock)     
        elif args.lingen.use_mamba:
            from src.model.lingen_blocks import LinGenBlock
            super().__init__(args, LinGenBlock)
        else:
            super().__init__(args, DiTTransformerBlock)

        # self.tok_embeddings, and self.pos_embeddings
        # were created in default in transformer.py
        # we kept them for legacy reason and remove them in dit_ transformer. py
        if self.tok_embeddings is not None:
            del self.tok_embeddings
            self.tok_embeddings = None
        if self. remove_positional_embeddings or self.factorize_positional_embeddings:
            del self-pos_embeddings
            self.pos_embeddings = None
        # create the positional embeddings for dit_ transformer
        self.pe_layers = args.pe_layers
        self.bundled_fsdp = args.get("bundled_fsdp", False)
        if not self.factorized_attn:
            if not self.bundled_fsdp or args.pe_layers == "first_layer_only":
                self.pe_blocks = PositionalEmbeddingsBlock(args)
            else:
                # if using bundled_fsdp, we need to create PE inside transformer 
                # blocks directly for all_layer_diff_pe
                self.pe_blocks = None  
        else:
            self.pe_blocks = FactorizedPositionalEmbeddingsBlock(args)

        # Get temporal patch_size from _embedder to decide size of
        # output layer
        self.out_channels = args.output_channels
        self.patch_size = args.patch_size
        self.x_embedder = build_patch_embedder(args)
        self.temporal_patch_size = args.temporal_patch_size
        self.use_y_embedder = args.use_y_embedder
        self.use_y_dummy = args.get("use_y_dummy", False)

        if args.x_cat_embedder:
            self.x_cat_embedder = build_patch_embedder(args)
        else:
            self.x_cat_embedder = None

        self.t_embedder = TimestepEmbedder(
            args.dim,
            args.frequency_embedding_dim, 
            init_args=InitArgs(fixed_std=args.timestep_init_std),
            use_sequence_parallel=args.sequence_parallel,
            fuse_sequence_parallel=args.fuse_sequence_parallel,
            non_linearity=args.timestep_non_linearity, 
            dropout=args.dropout, 
            use_te=False, 
            fc_bias=args.fe_bias,
        )

        init_depth = None 
        self.t_block_non_linearity = {
            "relu": F.relu,
            "gelu": F.gelu,
            "swiglu": None,
            "approx_gelu": partial(F.gelu, approximate="tanh"),
            "selu": lambda x: F.relu(x) ** 2,
            "silu": F.silu,
            # "mish"; F.mish,
            # "swish": swish,
        }[args.t_block_non_linearity]
        self.t_block = ColumnParallelLinear(
            input_size=args.dim, 
            output_size=args.dim * 6, 
            bias=args.t_block_bias,
            gather_output=True,
            init_method=get_init_fn(args.init, args.dim, init_depth),
            params_dtype=torch.get_default_dtype(),
        )

        context_dims = args.context_dim
        if not isinstance(context_dims, (list, ListConfig)):
            context_dims = [context_dims]
        
        self.layernorm_context = args.layernorm_context
        if self.layernorm_context:
            logger.warning(
                f" {args.layernorm_context=} - Applying layernorm on context is not advised. 
                Only apply it if you are sure about it."
            )
            self.context_norm = nn.ModuleList()
            for context_dim in context_dims:
                self.context_norm.append(
                    build_norm_fn(
                        args.norm_type,
                        context_dim,
                        args.layernorm_context_eps, 
                        args.lagernorm_context_affine,
                        sequence_parallel=args.sequence_parallel,
                    )
                )
        if self.use_y_embedder:
            self.y_embedders = nn.ModuleList(
                [
                    ContextEmbedder( 
                        in_dim=context_dim, 
                        out_dim=args.dim,
                        init_args=args.init,
                        use_sequence_parallel=args.sequence_parallel, 
                        fuse_sequence_parallel=args.fuse_sequence_parallel,
                        non_linearity=args.context_non_linearity,
                        dropout=args.context_embedder_dropout,
                        use_te=False, 
                        fc_bias=args.fc_bias,
                        norn_type=args.norm_type,
                        norm_affine=args.norm_affine, 
                        norm_eps=args.norm_eps,
                        context_norm=args.context_norm,
                    )
                    for context_dim in context_dims
                ]
            )

        self.use_parallel_cross_attn = args.use_parallel_cross_attn
        self.layernorm_y = args.layernorm_y
        if self.layernorm_y:
            self.y_norm = nn.ModuleList(
                [
                    build_norm_fn(
                        args.norm_type,
                        args.dim,
                        args.layernorm_y_eps,
                        args.layernorm_y_affine,
                        sequence_parallel=args.sequence_parallel,
                    )
                    for _ in context_dims
                ]
            )
                
        self.final_layer_scale_shift_table = Parameter(
            torch.randn(2, args.dim) / args.dim**0.5,
        )
        self.reuse_last_y_embedder = args.get("reuse_last_y_embedder", False)
        if args.freeze_model:
            self.freeze()
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            logger.info(f"Freezing {name}")
    
    def unpatchify(self, x, height=None, width=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, w, C)
        """
        c = self.out_channels
        p = self.patch_size
        tp = self.temporal_patch_size
        if height is None or width is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = height, width
        assert (
            h * w == x.shape[1]
        ), f"Height {h} * width {w} is not equal to x shape {x.shape[1]}"

        x = x.reshape(shape=(x.shape [0], h, w, tp, p, p, c))
        x = torch.einsum("nhwtpqc->ntchpva", x)
        return x.reshape(shape=(x.shape[0] * tp, c, h * p, w * p))

    def embed_y(self, y):
        if not isinstance(y, list):
            y = [y]
        y_embs = []

        if not self.reuse_last_y_embedder:
            y_embedder_iter = zip(self. y_embedders, y)
        else:
            # Reuse the last y embedder for extra y's
            y_embedder_iter = itertools.zip_longest(
                self.y_embedders, y, fillvalue=self.y_embedders[-1]
            )
        
        for i, (y_embedder, yi) in enumerate(y_embedder_iter):
            if self.layernorm_context:
                yi = self.context_norm[i](yi)
            y_embed = y_embedder(yi)
            y_embs.append(y_embed)
        
        if self.layernorm_y:
            y_embs = [norm(y_emb) for y_emb, norm in zip(y_embs, self.y_norm)]
        if self.use_parallel_cross_attn:
            y_embs = torch.stack(y_embs)
        else:
            y_embs = torch.cat(y_embs, dim=1)[None]
        return y_embs
    
    def maybe_apply_x_channels(self, x, cond):

        if CondUsageType.X_CHANNEL in cond:
            img_input = cond[CondUsageType.X_CHANNEL]
            if isinstance(img_input, list):
                img_input = torch. cat (img_input, dim=2)
            if img_input.ndim == 4:
                img_input = img_input.unsqueeze(1) 
            x = torch.cat ([x, img_input], dim=2)   
        return x

    def maybe_apply_x_add(self, x, cond):
        if CondUsageType.X_ADD in cond:
            img_input = cond[CondUsageType.X_ADD]
            if isinstance(img_input, list):
                img_input = torch.cat(img_input, dim=2)
            # x = torch.cat([x, img_input], dim=2)
            x = x + img_input
        return x

    def maybe_apply_x_cat(self, h, cond, bs, num_frames):
        if CondUsageType.X_CAT not in cond:
            return h, num_frames
        x_cats = cond[CondUsageType.X_CAT]
        if not isinstance(x_cats, Sequence):
            x_cat = x_cats
            num_x_cats = 1
        else:
            x_cat = torch.cat(x_cats, dim=1)
            num_x_cats = len(x_cats)
        if x_cat.ndim == 4:
            x_cat = x_cat.unsqueeze(1)
        if self.x_cat_embedder is None:
            h_cond = self.x_embedder(x_cat)
        else:
            h_cond = self.x_cat_embedder(x_cat)
        h_cond = rearrange(h_cond, "(b f) d h w -> b (f h w) d", b=bs)
        h = torch.cat([h, h_cond], dim=1)
        return h, num_x_cats + num_frames

    def loop_all_layers(
        self, 
        h,
        y,
        t0, 
        mask, 
        attn_bias, 
        slen, 
        batch_size, 
        height, 
        width, 
        frames, 
        cache, 
        is_first_microbatch, 
        is_last_microbatch, 
        recompute_attn, 
        recompute_fc1_fc3,
        step_idx, 
        mlp_id,
        return_features, 
        bs,
        N,
        ori_shape: List[int],
        pos_emb_scale: Optional[int] = None,
    ):

        def pe_calc(h, i, pos_emb_scale: Optional[int] = None):
            temporal_pos_emb = None
            if not self.factorized_attn and self.pe_blocks is not None:
                h = self.pe_blocks(
                    h, batch_size, height, width, frames, i, pos_emb_scale=pos_emb_scale
                )
            elif self.pe_blocks is not None: 
                h, spatial_pos_emb, temporal_pos_emb = self.pe_blocks(
                    h, batch_size, height, width, frames, i, pos_emb_scale=pos_emb_scale
                )
                if spatial_pos_emb is not None:
                    # add spatial pos_emb
                    h = h + spatial_pos_emb
            return h, temporal_pos_emb
        
        # Note: [Naman] This is to rate limit cpu thread from launching too many kernels
        # and get ahead as that increases gpu reserved memory and makes the run slower
        fwd_limit_queue = collections.deque()

        features_history = [] if return_features else None

        temporal_pos_emb = None

        for i, layer in enumerate(self.layers):
            if not self.bundled_fsdp or self.pe_layers == "first_layer_only":
                h, temporal_pos_emb = pe_calc(h, i, pos_emb_scale=pos_emb_scale)
            
            if self.pe_blocks is not None:
                freqs_cis = self.pe_blocks.get_rope_embeddings(
                    cache, slen, height, width, frames
                )
            else:
                freqs_cis = None
            
            if len(fwd_limit_queue) >= 2:
                event = fwd_limit_queue.popleft()
                event.synchronize()
            
            h, layer_cache = layer(
                h,
                y,
                te, 
                mask, 
                attn_bias, 
                fregs_cis,
                slen, 
                cache=cache,
                is_first_microbatch=is_first_microbatch,
                is_last_microbatch=is_last_microbatch, 
                recompute_attn=recompute_attn,
                recompute_fc1__fc3=recompute_fc1_fc3,
                step_idx=step_idx,
                mip_id=mlp_id,
                ori_shape=ori_shape,
                temporal_pos_emb=temporal_pos_emb, 
                long_context_buffers=(
                    self.long_context_shared_buffers if self.long_context_mode else None
                ),
                long_context_ffn_swiglu=(
                    self._long_context_ffn_swiglu
                    if self.long_context_cuda_graph_mode 
                    else None
                ),
                long_context_attn=(
                    self._long_context_attn
                    if self.long_context_cuda_graph_mode
                    else None
                ),
                long_context_cross_attn=(
                    self._long_context_cross_attn 
                    if self. long_context_cuda_graph_mode 
                    else None
                ),
                layer_id=i,
            )
            if return_features:
                features_history.append(h.view(bs, N, self.dim))
            
            if hasattr(layer, "logs") and layer.logs is not None:
                for k, v in layer.logs.items():
                    new_k = f"layer_{layer.layer_1d:03d}.{k}"
                    log_accumulation(self.logs, new_k, v)
            
            event = torch.cuda.Event ()
            event.record()
            fwd_limit_queue.append(event)
            
        return h, features_history
    
    def calc_e_diff_mlp_id(self, timestep):
        # use timestep to decide to use which MLP to train this batch
        if self.routing_method == "majority":
            if timestep.ndim == 2:
                F = timestep.shape[1]
                timestep = timestep[:, F // 2]
            mlp_id = (
                torch.floor(timestep / self.timesteps * self.n_ediff_mlps)
                .mode()
                .values.to(torch.int32)
                .item()
            )
        elif self.routing_method == "first_example":
            mlp_id = (
                torch.floor(timestep[0] / self.timesteps * self.n_ediff_mlps)
                .to(torch.int32)
                .item()
            )
        elif self.routing_method == "first_example_with_probablity":
            mlp_id = (
                torch.floor(timestep[0] / self.timesteps * self.n_ediff_mlps)
                .to(torch.int32)
                .item()
            )
            expert_probability = self.expert_probability
            non_expert_probability = (1.0 - expert_probability) / (
                self.n_ediff_mlps - 1
            )

            probs = torch.full((self.n_ediff_mlps,), non_expert_probability)
            probs[mlp_id] = expert_probability

            mlp_id = torch.multinomial(probs, 1).to(torch.int32).item()
        else:
            raise NotImplementedError(
                f"routing_method {self.routing_method} not implemented"
            )
        return mlp_id
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond: Dict[CondUsageType, Any],
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor,int]]] = None,
        pipeline_parallel_input_tensor: Optional[torch.Tensor] = None,
        is_first_microbatch: Union[bool, None] = None,
        is_last_microbatch: Union[bool, None] = True,
        recompute_attn: Optional[bool] = None,
        recompute_fc1_fc3: Optional[bool] = None,
        step_idx: int = -1,
        precomputed_attn_bias=None,
        add_to_timestep: Optional[torch.Tensor] = None,
        return_features: bool = False,
        pos_emb_scale: Optional[float] = None,
    ):
        x = self.maybe_apply_x_channels(x, cond)
        x = self.maybe_apply_x_add(x, cond)
        bs, frames, in_channels, in_height, in_width = x.shape

        if self.use_y_dummy:
            y = torch.zeros((1, x.shape[0], 1, self.dim)).to(x.device).to(x.dtype)
        elif self.use_y_embedder:
            y = cond[CondUsageType.CROSS_ATTN]
        else:
            y = cond
        h = self.x_embedder(x)      # (B f) x c x h x w

        # getting original shape of h in a format that is
        # compatible for window/factorized attention
        bs_n_frames, ch_size, height, width = h.shape

        ori_shape = [bs, bs_n_frames // bs, height, width, ch_size]

        h = rearrange(h, "(b f) d h w - b (f h w) d", b=bs)
        original_N = h.shape[1]     # original sequence length
        h, frames_after_x_cat = self.maybe_apply_x_cat(h, cond, bs, frames)
        N = h.shape [1]             # sequence length
        ori_shape[1] = frames_after_x_cat

        if mpu.get_context_parallel_world_size() > 1:
            assert (
                N % mpu.get_context_parallel_world_size() == 0
            ), f"seq_length: {N} must be divisible by cp_size: {mpu.get_context_parallel_world_size()}"
            N = N // mpu.get_context_parallel_worldsize()
        
        attn_bias = None
        if precomputed_attn_bias is not None:
            attn_bias = precomputed_attn_bias

        # embedding layer
        if (
            mpu.get_pipeline_model_parallel_world_size() == 1
            or mpu.is_pipeline_first_stage()
        ):
            if self.logs is not None:
                log_accumulation(self.logs, "input", norm_(h))
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_te:
                h = h.permute((1, 0, 2)).contiguous()
        
            # select CP seq chunk
            if mpu.get_context_parallel_world_size() > 1:
                assert h.dim() == 3, f"expecting shape of [B, N, D] but got {h.shape=}"
                h = dit_select_cp_seq_chunk(h, 1) # seq_dim = 1
            
            # Flatten the first two dims so that we're dealing with a batch of
            # individual tokens rather than a batch of sequences of tokens.
            # It's required for sequence parallelism, but we just always do it.
            h = h. reshape(N * bs, self.dim)
            # sequence parallel is default=true for TE
            if self.sequence_parallel:
                h = scatter_to_sequence_parallel_region(h)
            # When use balanced PP and only have embedding on first PP rank,
            # need to operate on imput so that we can dellocate it in pipeline 
            # to not encouter tensor. base is not None issue
            if len(self. layers) == 0:
                h = h * 1.0
        
        else:
            assert pipeline_parallel_input_tensor is not None
            h = pipeline_parallel_input_tensor

        if mpu.get_context_parallel_world_size() > 1:
            if timestep.dim() == 2: # (B, F)
                timestep = dit_select_cp_seq_chunk(timestep, 1) # seq_dim = 1
        
        t = self.t_embedder(timestep) # B → B D

        mlp_id = self.calc_e_diff_mlp_id(timestep)
        
        if add_to_timestep is not None:
            t0 = t + add_to_timestep
        else:
            t0 = t
        
        t0 = self.t_block_non_linearity(t0)
        t0 = self.t_block(t0)   # BD -> B 6D
        
        # y.shape = P x [B N E_i] (P=#encoders, E_i=dim of context i, N=context size)
        if self.use_y_embedder:
            y = self.embed_y(y)
        # if parallel attention: y.shape = [P B N D]
        # else: y.shape = [1 B (P N) D]
        
        if mpu.get_context_parallel_world_size() > 1 and self.dit_context_parallel_text: 
            # assert y.dim() == 3, f"expecting shape of [B, N, D] but got. (y.shape=}•
            assert y.dim() == 4, f"expecting shape of [P B N D] but got {y.shape=}"
            y = dit_select_cp_seq_chunk(y, 2) # seq_dim = 2
        
        y = rearrange(y, "p b l d → p (b l) d")
        # Don't use causal mask
        mask = None
        
        # transformer layers
        if self.use_fp8 and self.use_te:
            import transformer_engine
        
            context = transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_fp8,
                fp8_recipe=self.fp8_recipe,
                fp8_group=self.fp8_group,
            )
        else:
            context = nullcontext()
        if self.dit_sequence_parallel:
            h = h.reshape(bs, N, self.dim)
            h = h.transpose(0, 1)
            h = scatter_to_sequence_parallel_region(h)
            N = N // mpu.get_tensor_model_parallel_world_size()
            h = h.reshape((N * bs), self.dim)
    
            t0 = copy_to_tensor_model_parallel_region(t0)
    
        with context:
            h, features_history = self.loop_all_layers(
                h,
                y,
                t0, 
                mask, 
                attn_bias,
                N,
                bs,
                height, 
                width, 
                frames_after_x_cat, 
                cache, 
                is_first_microbatch, 
                is_last_microbatch, 
                recompute_attn, 
                recompute_fc1_fc3,
                step_idx,
                mlp_id,
                return_features, 
                bs,
                N,
                ori_shape=ori_shape,
                pos_emb_scale=pos_emb_scale,
            )

        if self.dit_sequence_parallel:
            h = gather_from_sequence_parallel_region(h, False)
            N = N * mpu.get_tensor_model_parallel_world_size()
            h = h.reshape(N, bs, h.shape[-1])
            h = h.transpose(0, 1)
            h = h.reshape(bs * N, self.dim)
        
        if (
            mpu.get_pipeline_model_parallel_world_size() == 1 
            or mpu.is_pipeline_last_stage()
        ):
            if len(self.layers) == 0:
                h = _Identity.apply(h)
            # if sequence parallel, gather outputs
            if self.sequence_parallel:
                h = gather_from_sequence_parallel_region(h, False)
            # Unflatten the first dimension in order to recover the original
            # batch grouped by sequences of tokens.
            if self.use_te:
                h = h.view(N, bs, self.dim).permute(1, 0, 2).contiguous()
            else:
                h = h.view(bs, N, self.dim)
            if self.logs is not None:
                log_accumulation(self.logs, "last_layer.output", norm_(h))
            
            # final_ layer from metavit
            if t.dim() == 2:
                shift, scale = (
                    self.final_layer_scale_shift_table[None] + t[:, None]
                ).chunk (2, dim=1)
            else:
                shift, scale = (
                    self.final_layer_scale_shift_table[None, None] + t[:, :, None]
                ).chunk(2, dim=2)
            
            # output layer
            if self.norm is not None and self.use_final_norm:
                h = self.norm(h)
            
            h = t2i_modulate(h, shift, scale)
            
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.logs is not None:
                log_accumulation(self. logs, "last_layer.normalised_output", norm_(h))
            # if loss parallel, MP 1, and the output only convers a 1/MP of the vocab
            if self.loss_parallel:
                output = linear_with_grad_accumulation_and_async_allreduce(
                    h,
                    self.output.weight,
                    bias=self.output.bias,
                    gradient_accumulation_fusion=False,
                    async_grad_allreduce=True,
                    sequence_parallel_enabled=False,
                )
                assert output.shape == (
                    bs, 
                    N,
                    self.vocab_size // self.tensor_model_parallel_size,
                )
            else:
                output = self.output(h)
            if self.logs is not None:
                log_accumulation(self.logs, "output", norm_(output))
            output = output.float()
        else:
            output = h
        
        # remerge/cp loss
        if mpu.get_context_parallel_world_size() > 1:
            with torch.no_grad():
                # h shape is [bs, N, d]
                output_list = [
                    torch.empty_like(output) 
                    for _ in range(mpu.get_context_parallel_world_size())
                ]
                torch.distributed.all_gather(
                    output_list, output, mpu.get_context_parallel_group()
                )
            output_list[mpu.get_context_parallel_rank()] = output
            output = torch.cat(output_list, dim=1)  # sea_dim = 1
            N = N * mpu.get_context_parallel_world_size()
        
        if original_N != N:
            output = output[:, -original_N:]

        output = rearrange(output, "b (f hw) c → (b f) hw c", f=frames)
        output = self.unpatchify(output, height, width)

        output = rearrange(output, "(b f) c h w -> b f c h w", b=bs)
    
        if return_features:
            return output, features_history

        return output