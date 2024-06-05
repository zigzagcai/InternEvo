import torch
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.parallel.comm.utils import all_gather_raw, reduce_scatter_raw

from .utils import RingComm, update_out_and_lse


def zigzag_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):
    assert causal is True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2

    def forward(q, k, v, causal):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    def get_next_kv_idx(prev_idx) -> int:
        if prev_idx == 0:
            return comm.world_size - 1
        else:
            return prev_idx - 1

    def _head_forward(q, full_k, full_v):
        out = None
        lse = None

        for step in range(comm.world_size):

            if step == 0:
                cur_kv_idx = comm.rank
                block_out, block_lse = forward(
                    q,
                    full_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    full_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    causal=True,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            elif step <= comm.rank:
                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k0 = full_k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                v0 = full_v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k = full_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                v = full_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                q1 = q[:, block_seq_len:]
                block_out, block_lse = forward(q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )

        return out, lse

    head_overlap_enable = gpc.config.ring_attn_overlap.get("enable", False)
    if head_overlap_enable:
        head_chunks = gpc.config.ring_attn_overlap.get("head_chunks", 1) 
        assert head_chunks > 1, "when enables the head overlap, the head chunks should be > 1."
        assert k.shape[-2] % head_chunks == 0, "the number of head should be divided by the head chunks."
    else:
        head_chunks = 1
    
    head_step = k.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)

    outs = []
    lses = []

    for i in range(head_chunks):
        if i == 0:
            k_cur, handle_k_cur = all_gather_raw(k_splits[i], process_group, async_op=True, gather_dim=1)
            v_cur, handle_v_cur = all_gather_raw(v_splits[i], process_group, async_op=True, gather_dim=1)
            handle_k_cur.wait()
            handle_v_cur.wait()
        else:
            handle_k_next.wait()
            handle_v_next.wait()
            k_cur = k_next
            v_cur = v_next

        if i != head_chunks - 1:
            k_next, handle_k_next = all_gather_raw(k_splits[i + 1], process_group, async_op=True, gather_dim=1)
            v_next, handle_v_next = all_gather_raw(v_splits[i + 1], process_group, async_op=True, gather_dim=1)

        out, lse = _head_forward(q[..., i * head_step : (i + 1) * head_step, :], k_cur, v_cur)
        
        lse = lse.squeeze(dim=-1).transpose(1, 2)

        outs.append(out)
        lses.append(lse)

    out = torch.cat(outs, dim=-2).to(q.dtype)
    lse = torch.cat(lses, dim=-2)
    return out, lse


def create_buffer(tensor, head_chunks, dim):
    buffer_shape = list(tensor.shape)
    buffer_shape[dim] //= head_chunks
    return torch.empty(buffer_shape, dtype=tensor.dtype, device=tensor.device)


def zigzag_ring_flash_attn_backward_full_kv(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):
    if gpc.get_global_rank() == 0:
        print("full_kv_zigzag_with_full_dkv = False", flush=True)
    assert causal is True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    head_overlap_enable = gpc.config.ring_attn_overlap.get("enable", False)
    if head_overlap_enable:
        head_chunks = gpc.config.ring_attn_overlap.get("head_chunks", 1) 
        assert head_chunks > 1, "when enables the head overlap, the head chunks should be > 1."
        assert k.shape[-2] % head_chunks == 0, "the number of head should be divided by the head chunks."
    else:
        head_chunks = 1
    
    head_step = k.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = create_buffer(q, head_chunks=head_chunks, dim=-2)
    dk_buffer = create_buffer(k, head_chunks=head_chunks, dim=-2)
    dv_buffer = create_buffer(v, head_chunks=head_chunks, dim=-2)

    def get_next_kv_idx(prev_idx) -> int:
        if prev_idx == 0:
            return kv_comm.world_size - 1
        else:
            return prev_idx - 1

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal,
        )

    def _head_backward(head_dout, head_q, head_k, head_v, head_out, head_softmax_lse):

        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None
        dk_comm_buffer, dv_comm_buffer = None, None

        for step in range(kv_comm.world_size):
            if step == 0:

                cur_kv_idx = kv_comm.rank
                backward(
                    head_dout,
                    head_q,
                    head_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    head_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    head_out,
                    head_softmax_lse,
                    causal=True,
                )
                dq = dq_buffer.to(torch.float32)
                dk = dk_buffer.to(torch.float32)
                dv = dv_buffer.to(torch.float32)
            else:
                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                if step <= kv_comm.rank:
                    k0 = head_k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                    v0 = head_v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                    backward(head_dout, head_q, k0, v0, head_out, head_softmax_lse, causal=False)
                    dq += dq_buffer
                else:
                    k = head_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                    v = head_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]

                    dout1 = head_dout.chunk(2, dim=1)[1]
                    q1 = head_q.chunk(2, dim=1)[1]
                    out1 = head_out.chunk(2, dim=1)[1]
                    softmax_lse1 = head_softmax_lse.chunk(2, dim=2)[1].contiguous()
                    backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)

                    # always use the first half in dq_buffer.
                    dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]  # pylint: disable=E1137

                d_kv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv

                if step <= kv_comm.rank:
                    dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]  # pylint: disable=E1137
                    dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]  # pylint: disable=E1137
                else:
                    dk += dk_buffer
                    dv += dv_buffer

            next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
            next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
            d_kv_comm.commit()

        d_kv_comm.wait()

        return dq, next_dk, next_dv

    dqs, next_dks, next_dvs = [], [], []

    for i in range(head_chunks):
        if i == 0:
            # all gather the first part of k_splits ,v_splits
            k_cur, handle_k_cur = all_gather_raw(k_splits[i], process_group, async_op=True, gather_dim=1)
            v_cur, handle_v_cur = all_gather_raw(v_splits[i], process_group, async_op=True, gather_dim=1)
            handle_k_cur.wait()
            handle_v_cur.wait()
        else:
            handle_k_next.wait()
            handle_v_next.wait()
            k_cur = k_next
            v_cur = v_next

        # all gather the next part of k_splits, v_splits
        if i != head_chunks - 1:
            k_next, handle_k_next = all_gather_raw(k_splits[i + 1], process_group, async_op=True, gather_dim=1)
            v_next, handle_v_next = all_gather_raw(v_splits[i + 1], process_group, async_op=True, gather_dim=1)

        dq, next_dk, next_dv = _head_backward(
            dout[..., i * head_step : (i + 1) * head_step, :],
            q[..., i * head_step : (i + 1) * head_step, :],
            k_cur,
            v_cur,
            out[..., i * head_step : (i + 1) * head_step, :],
            softmax_lse[..., i * head_step : (i + 1) * head_step, :],
        )

        dqs.append(dq)
        next_dks.append(next_dk)
        next_dvs.append(next_dv)

    dq = torch.cat(dqs, dim=-2).to(q.dtype)
    next_dk = torch.cat(next_dks, dim=-2).to(q.dtype)
    next_dv = torch.cat(next_dvs, dim=-2).to(q.dtype)

    return dq, next_dk, next_dv


def zigzag_ring_flash_attn_backward_full_kv_dkv(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):
    if gpc.get_global_rank() == 0:
        print("full_kv_zigzag_with_full_dkv = True", flush=True)
    assert causal is True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)

    head_overlap_enable = gpc.config.ring_attn_overlap.get("enable", False)
    if head_overlap_enable:
        head_chunks = gpc.config.ring_attn_overlap.get("head_chunks", 1) 
        assert head_chunks > 1, "when enables the head overlap, the head chunks should be > 1."
        assert k.shape[-2] % head_chunks == 0, "the number of head should be divided by the head chunks."
    else:
        head_chunks = 1
    
    head_step = k.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = create_buffer(q, head_chunks=head_chunks, dim=-2)
    dk_buffer = create_buffer(k, head_chunks=head_chunks, dim=-2)
    dv_buffer = create_buffer(v, head_chunks=head_chunks, dim=-2)

    def get_next_kv_idx(prev_idx) -> int:
        if prev_idx == 0:
            return kv_comm.world_size - 1
        else:
            return prev_idx - 1

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal,
        )

    def _head_backward(head_dout, head_q, head_k, head_v, head_out, head_softmax_lse):

        full_dk = torch.zeros_like(head_k, dtype=torch.float32)
        full_dv = torch.zeros_like(head_v, dtype=torch.float32)
        dq, dk, dv = None, None, None

        for step in range(kv_comm.world_size):

            if step == 0:
                cur_kv_idx = kv_comm.rank
                backward(
                    head_dout,
                    head_q,
                    head_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    head_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    head_out,
                    head_softmax_lse,
                    causal=True,
                )
                dq = dq_buffer.to(torch.float32)
                dk = dk_buffer.to(torch.float32)
                dv = dv_buffer.to(torch.float32)
                full_dk[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len] += dk
                full_dv[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len] += dv
            else:
                if step <= kv_comm.rank:
                    cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                    k0 = head_k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                    v0 = head_v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                    backward(head_dout, head_q, k0, v0, head_out, head_softmax_lse, causal=False)
                    dq += dq_buffer
                else:
                    cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                    k = head_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                    v = head_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                    dout1 = head_dout.chunk(2, dim=1)[1]
                    q1 = head_q.chunk(2, dim=1)[1]
                    out1 = head_out.chunk(2, dim=1)[1]
                    softmax_lse1 = head_softmax_lse.chunk(2, dim=2)[1].contiguous()
                    backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                    # always use the first half in dq_buffer.
                    dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]  # pylint: disable=E1137

                if step <= kv_comm.rank:
                    full_dk[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len] += dk_buffer[
                        :, :block_seq_len
                    ].to(torch.float32)

                    full_dv[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len] += dv_buffer[
                        :, :block_seq_len
                    ].to(torch.float32)
                else:
                    full_dk[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len] += dk_buffer.to(
                        torch.float32
                    )
                    full_dv[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len] += dv_buffer.to(
                        torch.float32
                    )

        # reduce-scatter dk and kv
        dk, dk_rs_handles = reduce_scatter_raw(full_dk, process_group, async_op=True, reduce_dim=1)
        dv, dv_rs_handles = reduce_scatter_raw(full_dv, process_group, async_op=True, reduce_dim=1)
        return dq, dk, dv, dk_rs_handles, dv_rs_handles

    dqs, dks, dvs = [], [], []
    dk_rs_handles, dv_rs_handles = [], []

    for i in range(head_chunks):
        if i == 0:
            # all gather the first part of k_splits ,v_splits
            k_cur, handle_k_cur = all_gather_raw(k_splits[i], process_group, async_op=True, gather_dim=1)
            v_cur, handle_v_cur = all_gather_raw(v_splits[i], process_group, async_op=True, gather_dim=1)
            handle_k_cur.wait()
            handle_v_cur.wait()
        else:
            handle_k_next.wait()
            handle_v_next.wait()
            k_cur = k_next
            v_cur = v_next

        if i != head_chunks - 1:
            k_next, handle_k_next = all_gather_raw(k_splits[i + 1], process_group, async_op=True, gather_dim=1)
            v_next, handle_v_next = all_gather_raw(v_splits[i + 1], process_group, async_op=True, gather_dim=1)

        dq, dk, dv, dk_handle, dv_handle = _head_backward(
            dout[..., i * head_step : (i + 1) * head_step, :],
            q[..., i * head_step : (i + 1) * head_step, :],
            k_cur,
            v_cur,
            out[..., i * head_step : (i + 1) * head_step, :],
            softmax_lse[:, i * head_step : (i + 1) * head_step, :],
        )

        dqs.append(dq)
        dks.append(dk)
        dvs.append(dv)
        dk_rs_handles.append(dk_handle)
        dv_rs_handles.append(dv_handle)

    dq_final = torch.concat(dqs, dim=-2)
    for dk_rs_handle in dk_rs_handles:
        dk_rs_handle.wait()
    dk_final = torch.concat(dks, dim=-2)
    for dv_rs_handle in dv_rs_handles:
        dv_rs_handle.wait()
    dv_final = torch.concat(dvs, dim=-2)

    return dq_final.to(q.dtype), dk_final.to(k.dtype), dv_final.to(v.dtype)


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    """ZigZagRingFlashAttnFunc"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        with_full_dkv,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()

        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group, q, k, v, softmax_scale=softmax_scale, dropout_p=dropout_p, causal=causal
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.with_full_dkv = with_full_dkv
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):  # pylint: disable=W0613
        q, k, v, out, softmax_lse = ctx.saved_tensors
        with_full_dkv = ctx.with_full_dkv

        backward_func = (
            zigzag_ring_flash_attn_backward_full_kv_dkv if with_full_dkv else zigzag_ring_flash_attn_backward_full_kv
        )

        dq, dk, dv = backward_func(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_qkvpacked_func_with_full_kv(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        gpc.config.get("full_kv_zigzag_with_full_dkv", False),  # TODO: pass by args.
        group,
    )


def zigzag_ring_flash_attn_kvpacked_func_with_full_kv(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        gpc.config.get("full_kv_zigzag_with_full_dkv", False),  # TODO: pass by args.
        group,
    )


def zigzag_ring_flash_attn_func_with_full_kv(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        gpc.config.get("full_kv_zigzag_with_full_dkv", False),  # TODO: pass by args.
        group,
    )
