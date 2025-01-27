from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union 
import torch

from src.args import InitArgs, ModelArgs, TrainerArgs 
from src.model.helpers import to_3tuple
from src.model.long_context_mode.long_context_modules import (
    LongContextAttentionCUDAGraph, 
    LongContextAttentionSharedBuffers, 
    LongContextCrossAttentionCUDAGraph, 
    LongContextFFNSwigluCUDAGraph,
)
from src.model.window_attention import WindowAttention3D 
from src.parallelism import parallel_state as mpu 
from src.parallelism.tensor_parallel import (
    copy_to_tensor_model_parallel_region, 
    gather_from_sequence_parallel_region, 
    scatter_to_sequence_parallel_region, 
)
from src.utils import norm_

logger = getLogger()

try:
    from xformers import ops as xops 
    from xformers.Ops import AttentionBias 
except (ImportError, ModuleNotFoundError):
    AttentionBias = Any
    logger.warn("xFormers AttentionBias op not found")

from src.model.dit_transformer_blocks import DiTTransformerBlock 
from src.model.mamba_blocks.mamba import Mamba 
from src.model.mamba_blocks.mamba2 import Mamba2

def t2i_modulate(x, shift, scale, seq_leading: bool = False):
    if scale.dim() == 4:
        B, F, N, D = scale.shape        # [B, F, 1, D]
        assert N == 1
        x_ = x.view(B, F, -1, D)
        x_ = x_ * (1 + scale) + shift
        if x.dim() == 2:
            return x_.view(-1, D)       # (B*f*h/p*w/p, D)
        else:
            return x_.view(B, -1, D)    # (B,f*h/p*w/p, D)
    if x.dim() == 2:
        B, N, D = scale. shape
        # NOTE this N isn't actually the N for seq_len.
        # it's the t chunk dim:
        assert N == 1
        scale = scale.squeeze(1)
        shift = shift.squeeze(1)
        if not seq_leading:
            x_ = x.view(B, -1, D)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        else:
            x_ = x.view(-1, B, D)
            scale = scale.unsqueeze(0)
            shift = shift.unsqueeze(0) 
        x_= x_ * (1 + scale) + shift 
        return x_.view(-1, D)
    else:
        return x * (1 + scale) + shift

def t2i_gate(x, gate, seq_leading: bool = False):
    if gate.dim() == 4:
        B, F, N, D = gate.shape
        assert N == 1
        x_ = x.view(B, F, -1, D)
        x_ = x_ * gate
        if x.dim() == 2:
            return x_.view(-1, D)
        else:
            return x_.view(B, -1, D)
        
    if x.dim() == 3:
        return gate * x
    else:
        B, N, D = gate.shape
        assert N == 1
        gate = gate.squeeze(1)
        if not seq_leading:
            x_ = x.view(B, -1, D)
            gate = gate.unsqueeze(1)
        else:
            x_ = x.view(-1, B, D)
            gate = gate.unsqueeze(0)
        x_ = x_ * gate
        return x_.view(-1, D)

class LinGenBlock(DiTTransformerBlock):
    def __init__(self, args: ModelArgs, layer_id: int, use_te: bool = True):
        super().__init__(args, layer_id, use_te=use_te)
        self.use_mamba = args.lingen.use_mamba
        self.mamba_state_size = args.lingen.mamba_state_size
        self.mamba_conv_size = args.lingen.mamba_conv_size
        self.side_attn = args.lingen.mamba_side_winattn
        self.mamba_scan = args.lingen.mamba_scan
        self.use_mamba2 = args.lingen.use_mamba2
        self.review_tokens = args.lingen.review_tokens

        # layers
        init_depth = {None: None, "current": layer_id + 1, "global": args.n_layers}

        self.layer_id = layer_id
        if self.use_mamba:
            if self.use_mamba2:
                logger.info(
                    f"Replacing attention with Mamba2, d_model={args.dim}"
                    f"at layer_id={layer_id}"
                )
                self.attention = Mamba2(
                    d_model = args.dim,
                    d_state = (
                        self.mamba_state_size 
                        if self.mamba_state_size is not None 
                        else 128
                    ),
                    d_conv = (
                        self.mamba_conv_size 
                        if self.mamba_conv_size is not None 
                        else 4
                    ),
                    layer_idx = layer_id,
                    device = torch.cuda.current_device(),
                    bidirectional = True,
                    scan_type = self.mamba_scan,
                    review_tokens = self.review_tokens,
                )
            else:
                ### Use Mamba1
                logger.info(
                    f"Replacing attention with Mamba, d_model={args.dim}"
                    f"at layer_id={layer_id}"
                )
                self.attention = Mamba(
                    d_model = args.dim,
                    d_state = (
                        self.mamba_state_size 
                        if self.mamba_state_size is not None 
                        else 16
                    ),
                    d_conv = (
                        self.mamba_conv_size 
                        if self.mamba_conv_size is not None 
                        else 4
                    ),
                    layer_idx = layer_id,
                    device = torch.cuda.current_device(),
                    bimamba_type = "v2",
                    scan_type = self.mamba_scan,
                )
            if self.side_attn:
                window_size = to_3tuple(args.window_size)
                ### Alternate shifting
                if layer_id % 2 == 0:
                    shift_size = (0, 0, 0)
                else:
                    shift_size = tuple([w // 2 for w in window_size])
                logger.info(
                    f"Using side attention with window_size={window_size}"
                    f"shift_size={shift_size} at layer_id={layer_id}"
                )
                self.side_attention = WindowAttention3D(
                    dim = args.dim,
                    n_heads = self.n_heads,
                    n_kv_heads = (
                        args.get("n_kv_heads_self_attn", None)
                        if args.get("n_kv_heads_self_attn", None)
                        else self.n_heads
                    ),
                    head_dim = self.head_dim,
                    window_size = window_size,
                    shift_size = shift_size,
                    use_relative_position_bias = args.win_attn_use_relative_position_bias,
                    attn_drop = args.dropout,
                    proj_drop = 0.0,
                    efficient_attn = args.efficient_attn,
                    use_sequence_parallel = args.sequence_parallel,
                    fuse_sequence_parallel = args.fuse_sequence_parallel,
                    init_args = args.init,
                    init_depth = init_depth[args.init.use_depth],
                    norm_type = args.norm_type,
                    qk_norm_type = args.qk_norm_type,
                    norm_eps = args.norm_eps,
                    use_qk_norm = args.use_qk_norm_self_attn,
                    qk_norm_aross_heads = args.qk_norm_aross_heads,
                    qk_norm_eps = args.qk_norm_eps,
                    qk_norm_affine = args.qk_norm_affine,
                    use_te = use_te,
                    recompute_attn = args.recompute_attn,
                    fc_bias = args.win_attn_fc_bias,
                    qkv_fc_bias = args.win_attn_qkv_fc_bias,
                    attn_impl = args.win_attn_impl,
                )
    
    def forward(
        self,
        x: torch.Tensor,
        cross_x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_bias: Optional[AttentionBias],
        freqs_cis: Optional[torch.Tensor],
        slen: int,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        is_first_microbatch: Union[bool, None] = None,
        is_last_microbatch: Union[bool, None] = True,
        recompute_fc1_fc3: Optional[bool] = None,
        recompute_attn: Optional[bool] = None,
        step_idx: int = -1,
        mlp_id: int = 0,
        ori_shape: Optional[Tuple[int, ...]] = None,
        temporal_pos_emb: Optional[torch.Tensor] = None,
        long_context_buffers: Optional[LongContextAttentionSharedBuffers] = None,
        long_context_ffn_swiglu: Optional[LongContextFFNSwigluCUDAGraph] = None,
        long_context_attn: Optional[LongContextAttentionCUDAGraph] = None,
        long_context_cross_attn: Optional[LongContextCrossAttentionCUDAGraph] = None,
        layer_id: Optional[int] = None,
    ):
        """
        x: (N * bs, self.dim)
        """

        assert x.dim() == 2, f"why does x have dim {x.dim()}?"

        bs = x.shape[0] // slen

        assert t.dim() == 2 or t.dim() == 3
        assert t.shape[0] == bs
        assert t.shape[-1] == 6 * x.shape[1]

        scale_shift_table = self.scale_shift_table
        if self.dit_sequence_parallel:
            scale_shift_table = copy_to_tensor_model_parallel_region(scale_shift_table)
        
        assert x.shape[1] == scale_shift_table.shape[1]
        if t.dim() == 2:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                scale_shift_table.view(1, 6, x.shape[1]) + t.view(bs, 6, x.shape[1])
            ).chunk(6, dim=1)
        else:
            num_t = t.shape[1]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                scale_shift_table.view(1, 1, 6, x.shape[1]) + 
                t.view(bs, num_t, 6, x.shape[1])
            ).chunk(6, dim=2)
        
        ### Apply positional embedding
        if self.pe_layers == "all_layer_diff_pe" and self.bundled_fsdp:
            height, width, frames = ori_shape[2], ori_shape[3], ori_shape[1]
            pos_embed = self.get_pos_embedding(
                bs, height, width, frames, x.device, x.dtype
            )
            x = x + pos_embed
        
        h = t2i_modulate(
            self.attention_norm(x, self.dit_sequence_parallel),
            shift_msa,
            scale_msa,
            seq_leading=self.dit_sequence_parallel,
        )

        if self.dit_sequence_parallel and not self.long_context_mode:
            h = gather_from_sequence_parallel_region(h, False)
            slen = slen * mpu.get_tensor_model_parallel_world_size()
            h = h.reshape(-1, bs, h.shape[-1])
            h = h.transpose(0, 1)
            h = h.reshape(-1, h.shape[-1])
        
        if self.use_mamba:
            if self.side_attn:
                h_side = self.side_attention(h, ori_shape)
            if self.use_mamba2:
                h_attn = self.attention(
                    h,
                    ori_shape=ori_shape,
                    seqlen = ori_shape[1] * ori_shape[2] * ori_shape[3],
                )
            else:
                ### Use Mamba1
                h = h.reshape(bs, -1, h.shape[-1])
                h_attn = self.attention(h, ori_shape)
                h_attn = h_attn.reshape(-1, h_attn.shape[-1])
            if self.side_attn:
                h_attn = h_attn + h_side
        else:
            h_attn, cache = self.attention(
                h,
                mask=None,
                attn_bias=None,
                freqs_cis=freqs_cis,
                slen=slen,
                cache=cache,
                is_first_microbatch=is_first_microbatch,
                is_last_microbatch=is_last_microbatch,
                recompute_attn=recompute_attn,
                logs=self.logs,
                ori_shape=ori_shape,
                cp_gather_kv=True,
                long_context_buffers=(
                    long_context_buffers if self.long_context_mode else None
                ),
                long_context_attn=(
                    long_context_attn if self.long_context_cuda_graph_mode else None
                ),
                layer_id=layer_id,
            )
        
        if self.dit_sequence_parallel and not self.long_context_mode:
            h_attn = h_attn.reshape(bs, -1, h_attn.shape[-1])
            h_attn = h_attn.transpose(0, 1)
            h_attn = scatter_to_sequence_parallel_region(h_attn)
            slen = slen // mpu.get_tensor_model_parallel_world_size()
            h_attn = h_attn.reshape(-1, h_attn.shape[-1])
        
        h = x + t2i_gate(
            self.rescale(h_attn),
            gate_msa,
            seq_leading=self.dit_sequence_parallel,
        )

        if self.dit_sequence_parallel and not self.long_context_mode:
            h_cross = gather_from_sequence_parallel_region(h, False)
            slen = slen * mpu.get_tensor_model_parallel_world_size()
            h_cross = h_cross.reshape(-1, bs, h_cross.shape[-1])
            h_cross = h_cross.transpose(0, 1)
            h_cross = h_cross.reshape(-1, h_cross.shape[-1])
        else:
            h_cross = h
        
        h_cross, _ = self.cross_attention(
            x=h_cross,
            mask=None,
            attn_bias=None,
            freqs_cis=None,
            slen=slen,
            cross_x=cross_x,
            cache=cache,
            is_first_microbatch=is_first_microbatch,
            is_last_microbatch=is_last_microbatch,
            recompute_attn=recompute_attn,
            logs=self.logs,
            cp_gather_kv=self.dit_context_parallel_text,
            long_context_buffers=(
                long_context_buffers if self.long_context_mode else None
            ),
            long_context_cross_attn=(
                long_context_cross_attn if self.long_context_cuda_graph_mode else None
            ),
            layer_id=layer_id,
        )

        if self.dit_sequence_parallel and not self.long_context_mode:
            h_cross = h_cross.reshape(bs, -1, h_cross.shape[-1])
            h_cross = h_cross.transpose(0, 1)
            h_cross = scatter_to_sequence_parallel_region(h_cross)
            slen = slen // mpu.get_tensor_model_parallel_world_size()
            h_cross = h_cross.reshape(-1, h_cross.shape[-1])
        
        h = h + self.rescale(h_cross)

        h_mod = t2i_modulate(
            self.ffn_norm(h, self.dit_sequence_parallel),
            shift_mlp,
            scale_mlp,
            seq_leading=self.dit_sequence_parallel,
        )

        if self.dit_sequence_parallel and not self.long_context_mode:
            h_mod = gather_from_sequence_parallel_region(h_mod, False)
            slen = slen * mpu.get_tensor_model_parallel_world_size()
        
        if self.n_ediff_mlps == 1:
            h_ff = self.feed_forward(
                h_mod,
                step_idx=step_idx,
                long_context_buffers=(
                    long_context_buffers if self.long_context_mode else None
                ),
                long_context_ffn_swiglu=(
                    long_context_ffn_swiglu if self.long_context_cuda_graph_mode else None
                ),
                layer_id=layer_id,
                slen=slen,
            )
        elif (self.ensemble_inference and not self.training) or self.ensemble_train:
            h_ff = torch.zeros_like(h)
            non_experts_weight = (1.0 - self.inference_expert_weight) / (self.n_ediff_mlps - 1)
            for ind, ffn in enumerate(self.feed_forward):
                h_ff_ind = ffn(h_mod, step_idx=step_idx)
                if ind == mlp_id:
                    weight = self.inference_expert_weight
                else:
                    weight = non_experts_weight
                h_ff += h_ff_ind * weight
        elif self.n_ediff_mlps > 1 and mlp_id < self.n_ediff_mlps:
            h_ff = self.feed_forward(mlp_id, h_mod, step_idx=step_idx)
        else:
            raise ValueError("Invalid mlp_id")
        
        if self.dit_sequence_parallel and not self.long_context_mode:
            h_ff = scatter_to_sequence_parallel_region
            slen = slen // mpu.get_tensor_model_parallel_world_size()
        
        h_ff = t2i_gate(
            self.rescale(h_ff),
            gate_mlp,
            seq_leading=self.dit_sequence_parallel,
        )

        out = h + h_ff

        if self.logs is not None:
            self.logs["x"] = norm_(x)
            if h_attn is not None:
                self.logs["attn_output"] = norm_(h_attn)
            if h_cross is not None:
                self.logs["cross_attn_output"] = norm_(h_cross)
        
        return out, cache