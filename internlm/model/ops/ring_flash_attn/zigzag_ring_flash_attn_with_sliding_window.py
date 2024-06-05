import torch
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
import torch.distributed

from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.parallel.comm.utils import all_gather_raw, reduce_scatter_raw

from .utils import RingComm, update_out_and_lse


def create_buffer(tensor, head_chunks, dim):
    buffer_shape = list(tensor.shape)
    buffer_shape[dim] //= head_chunks
    return torch.empty(buffer_shape, dtype=tensor.dtype, device=tensor.device)


def zigzag_ring_flash_attn_forward(
    ring_pg,
    p2p_pg,
    all_gather_pg,
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

    if gpc.get_global_rank() == 0:
        print("P2P + AllGATHER FORWARD.........", flush=True)

    assert causal is True, "zigzag ring is meaningless for causal=False"
    ring_comm = RingComm(ring_pg)
    p2p_comm = RingComm(p2p_pg)

    all_gather_ws = torch.distributed.get_world_size(all_gather_pg)
    all_gather_local_rank = torch.distributed.get_rank(all_gather_pg)

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
            return all_gather_ws - 1
        else:
            return prev_idx - 1

    def _head_first_window_forward(q, full_k, full_v):
        out = None
        lse = None

        for step in range(all_gather_ws):

            if step == 0:
                cur_kv_idx = all_gather_local_rank
                block_out, block_lse = forward(
                    q,
                    full_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    full_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len],
                    causal=True,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            elif step <= all_gather_local_rank:
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

    def _head_other_window_forward(out, lse, q, full_k, full_v, window_num_idx):

        if window_num_idx > p2p_comm.rank:

            for step in range(all_gather_ws):
                if step == 0:
                    cur_kv_idx = all_gather_local_rank
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
        else:
            for step in range(all_gather_ws):
                if step == 0:
                    cur_kv_idx = all_gather_local_rank
                else:
                    cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k0 = full_k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                v0 = full_v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        return out, lse

    head_overlap_enable = gpc.config.ring_attn_overlap.get("enable", False)
    if head_overlap_enable:
        head_chunks = gpc.config.ring_attn_overlap.get("head_chunks", 1)
        assert head_chunks > 1, "when enables the head overlap, the head chunks should be > 1."
        assert k.shape[-2] % head_chunks == 0, "the number of head should be divided by the head chunks."
    else:
        head_chunks = 1

    window_size = gpc.config.ring_attn_overlap.get("window_size", 1)
    window_num = ring_comm.world_size // window_size

    head_step = q.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)

    outs = []
    lses = []

    comm_stream = torch.cuda.Stream()
    comm_event = torch.cuda.Event()
    compute_event = torch.cuda.Event()

    for i in range(head_chunks):

        local_k = k_splits[i]
        local_v = v_splits[i]

        for j in range(window_num):

            if j == 0:
                full_k, _ = all_gather_raw(local_k, all_gather_pg, async_op=False, gather_dim=1)
                full_v, _ = all_gather_raw(local_v, all_gather_pg, async_op=False, gather_dim=1)
                torch.cuda.current_stream().record_event(compute_event)
            else:
                torch.cuda.current_stream().wait_event(comm_event)
                local_k = next_k
                local_v = next_v
                full_k = full_k_next
                full_v = full_v_next

            if j + 1 != window_num:
                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_event(compute_event)
                    next_k: torch.Tensor = p2p_comm.send_recv(local_k.contiguous())
                    next_v: torch.Tensor = p2p_comm.send_recv(local_v.contiguous())
                    p2p_comm.commit()
                    p2p_comm.wait()
                    full_k_next, _ = all_gather_raw(next_k, all_gather_pg, async_op=False, gather_dim=1)
                    full_v_next, _ = all_gather_raw(next_v, all_gather_pg, async_op=False, gather_dim=1)
                    comm_stream.record_event(comm_event)

            if j == 0:
                out, lse = _head_first_window_forward(q[..., i * head_step : (i + 1) * head_step, :], full_k, full_v)
            else:
                out, lse = _head_other_window_forward(
                    out, lse, q[..., i * head_step : (i + 1) * head_step, :], full_k, full_v, window_num_idx=j
                )

            torch.cuda.current_stream().record_event(compute_event)

        lse = lse.squeeze(dim=-1).transpose(1, 2)

        outs.append(out)
        lses.append(lse)

    out = torch.cat(outs, dim=-2).to(q.dtype)
    lse = torch.cat(lses, dim=-2)

    return out, lse


def zigzag_double_ring_flash_attn_forward(
    ring_pg,
    p2p_pg,
    local_p2p_pg,
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

    if gpc.get_global_rank() == 0:
        print("DOUBLE RING FORWARD.........", flush=True)

    assert causal is True, "zigzag ring is meaningless for causal=False"
    ring_comm = RingComm(ring_pg)
    p2p_comm = RingComm(p2p_pg)
    local_p2p_comm = RingComm(local_p2p_pg)

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

    def _head_first_window_forward(q, k, v):
        out = None
        lse = None

        for step in range(local_p2p_comm.world_size):

            if step + 1 != local_p2p_comm.world_size:
                next_k: torch.Tensor = local_p2p_comm.send_recv(k)
                next_v: torch.Tensor = local_p2p_comm.send_recv(v)
                local_p2p_comm.commit()

            if step == 0:
                block_out, block_lse = forward(
                    q,
                    k,
                    v,
                    causal=True,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            elif step <= local_p2p_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                q1 = q[:, block_seq_len:]
                block_out, block_lse = forward(q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )

            if step + 1 != local_p2p_comm.world_size:
                local_p2p_comm.wait()
                k = next_k
                v = next_v

        return out, lse

    def _head_other_window_forward(out, lse, q, k, v, window_num_idx):

        if window_num_idx > p2p_comm.rank:

            for step in range(local_p2p_comm.world_size):

                if step + 1 != local_p2p_comm.world_size:
                    next_k: torch.Tensor = local_p2p_comm.send_recv(k)
                    next_v: torch.Tensor = local_p2p_comm.send_recv(v)
                    local_p2p_comm.commit()

                q1 = q[:, block_seq_len:]
                block_out, block_lse = forward(q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )

                if step + 1 != local_p2p_comm.world_size:
                    local_p2p_comm.wait()
                    k = next_k
                    v = next_v
        else:
            for step in range(local_p2p_comm.world_size):

                if step + 1 != local_p2p_comm.world_size:
                    next_k: torch.Tensor = local_p2p_comm.send_recv(k)
                    next_v: torch.Tensor = local_p2p_comm.send_recv(v)
                    local_p2p_comm.commit()

                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

                if step + 1 != local_p2p_comm.world_size:
                    local_p2p_comm.wait()
                    k = next_k
                    v = next_v
        return out, lse

    head_overlap_enable = gpc.config.ring_attn_overlap.get("enable", False)
    if head_overlap_enable:
        head_chunks = gpc.config.ring_attn_overlap.get("head_chunks", 1)
        assert head_chunks > 1, "when enables the head overlap, the head chunks should be > 1."
        assert k.shape[-2] % head_chunks == 0, "the number of head should be divided by the head chunks."
    else:
        head_chunks = 1

    window_size = gpc.config.ring_attn_overlap.get("window_size", 1)
    window_num = ring_comm.world_size // window_size

    head_step = q.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)

    outs = []
    lses = []

    for i in range(head_chunks):

        local_k = k_splits[i]
        local_v = v_splits[i]

        for j in range(window_num):

            if j > 0:
                p2p_comm.wait()
                local_k = next_k
                local_v = next_v

            if j + 1 != window_num:
                next_k: torch.Tensor = p2p_comm.send_recv(local_k.contiguous())
                next_v: torch.Tensor = p2p_comm.send_recv(local_v.contiguous())
                p2p_comm.commit()

            if j == 0:
                # out, lse = _head_first_window_forward(q[..., i * head_step : (i + 1) * head_step, :], local_k, local_v)
                out, lse = _head_first_window_forward(q, local_k, local_v)
            else:
                out, lse = _head_other_window_forward(
                    out, lse, q[..., i * head_step : (i + 1) * head_step, :], local_k, local_v, window_num_idx=j
                )

        lse = lse.squeeze(dim=-1).transpose(1, 2)
        out = out.to(q.dtype)

        outs.append(out)
        lses.append(lse)

    out = torch.cat(outs, dim=-2).to(q.dtype).contiguous()
    lse = torch.cat(lses, dim=-2).contiguous()
    return out, lse


def zigzag_double_ring_flash_attn_backward(
    ring_pg,
    p2p_pg,
    local_p2p_pg,
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
        print("DOUBLE RING BACKWARD.........", flush=True)
    assert causal is True, "zigzag ring is meaningless for causal=False"

    ring_comm = RingComm(ring_pg)
    dkv_comm = RingComm(p2p_pg)
    kv_comm = RingComm(p2p_pg)
    local_kv_comm = RingComm(local_p2p_pg)
    local_dkv_comm = RingComm(local_p2p_pg)

    # print(f"xyt global rank = {gpc.get_global_rank()}, ring_pg = {torch.distributed.get_process_group_ranks(ring_pg)}, p2p_pg = {torch.distributed.get_process_group_ranks(p2p_pg)}, local_p2p_pg = {torch.distributed.get_process_group_ranks(local_p2p_pg)}", flush=True)

    head_overlap_enable = gpc.config.ring_attn_overlap.get("enable", False)
    if head_overlap_enable:
        head_chunks = gpc.config.ring_attn_overlap.get("head_chunks", 1)
        assert head_chunks > 1, "when enables the head overlap, the head chunks should be > 1."
        assert k.shape[-2] % head_chunks == 0, "the number of head should be divided by the head chunks."
    else:
        head_chunks = 1

    head_step = q.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = create_buffer(q, head_chunks=head_chunks, dim=-2)
    dk_buffer = create_buffer(k, head_chunks=head_chunks, dim=-2)
    dv_buffer = create_buffer(v, head_chunks=head_chunks, dim=-2)

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        softmax_lse = softmax_lse.contiguous()
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

    def _head_first_window_backward(dout, q, k, v, out, softmax_lse):

        dk_comm_buffer, dv_comm_buffer = None, None
        dq, dk, dv = None, None, None

        for step in range(local_kv_comm.world_size):
            if step + 1 != local_kv_comm.world_size:
                next_k = local_kv_comm.send_recv(k)
                next_v = local_kv_comm.send_recv(v)
                local_kv_comm.commit()

            if step == 0:
                backward(dout, q, k, v, out, softmax_lse, causal=True)
                dq = dq_buffer.to(torch.float32)
                dk = dk_buffer.to(torch.float32)
                dv = dv_buffer.to(torch.float32)
            else:
                if step <= local_kv_comm.rank:
                    k0 = k[:, :block_seq_len]
                    v0 = v[:, :block_seq_len]
                    backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                    dq += dq_buffer
                else:
                    dout1 = dout.chunk(2, dim=1)[1]
                    q1 = q.chunk(2, dim=1)[1]
                    out1 = out.chunk(2, dim=1)[1]
                    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
                    backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                    # always use the first half in dq_buffer.
                    dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

                local_dkv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv

                if step <= local_kv_comm.rank:
                    dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                    dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
                else:
                    dk += dk_buffer
                    dv += dv_buffer

            if step + 1 != local_kv_comm.world_size:
                local_kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
            next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
            local_dkv_comm.commit()

        local_dkv_comm.wait()

        return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

    def _head_other_window_backward(dout, q, k, v, dq, dk, dv, out, softmax_lse, window_num_idx):

        dk_comm_buffer, dv_comm_buffer = None, None

        if window_num_idx > kv_comm.rank:

            for step in range(local_kv_comm.world_size):

                if step + 1 != local_kv_comm.world_size:
                    next_k = local_kv_comm.send_recv(k)
                    next_v = local_kv_comm.send_recv(v)
                    local_kv_comm.commit()

                dout1 = dout.chunk(2, dim=1)[1]
                q1 = q.chunk(2, dim=1)[1]
                out1 = out.chunk(2, dim=1)[1]
                softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

                if step > 0:
                    local_dkv_comm.wait()
                    dk_comm_buffer, dv_comm_buffer = dk, dv
                    dk, dv = next_dk, next_dv

                dk += dk_buffer
                dv += dv_buffer

                if step + 1 != local_kv_comm.world_size:
                    local_kv_comm.wait()
                    k = next_k
                    v = next_v

                next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
                next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
                local_dkv_comm.commit()

            local_dkv_comm.wait()
        else:

            for step in range(local_kv_comm.world_size):

                if step + 1 != local_kv_comm.world_size:
                    next_k = local_kv_comm.send_recv(k)
                    next_v = local_kv_comm.send_recv(v)
                    local_kv_comm.commit()

                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer

                if step > 0:
                    local_dkv_comm.wait()
                    dk_comm_buffer, dv_comm_buffer = dk, dv
                    dk, dv = next_dk, next_dv

                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]

                if step + 1 != local_kv_comm.world_size:
                    local_kv_comm.wait()
                    k = next_k
                    v = next_v

                next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
                next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
                local_dkv_comm.commit()

            local_dkv_comm.wait()

        return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

    window_size = gpc.config.ring_attn_overlap.get("window_size", 1)
    window_num = ring_comm.world_size // window_size

    dqs, next_dks, next_dvs = [], [], []

    for i in range(head_chunks):

        local_k = k_splits[i]
        local_v = v_splits[i]

        for j in range(window_num):

            if j > 0:
                kv_comm.wait()
                local_k = next_k
                local_v = next_v

            if j + 1 != window_num:
                next_k: torch.Tensor = kv_comm.send_recv(local_k.contiguous())
                next_v: torch.Tensor = kv_comm.send_recv(local_v.contiguous())
                kv_comm.commit()

            if j > 0:
                dkv_comm.wait()
                dk = next_dk
                dv = next_dv

            if j == 0:
                dq, dk, dv = _head_first_window_backward(
                    dout[..., i * head_step : (i + 1) * head_step, :],
                    q[..., i * head_step : (i + 1) * head_step, :],
                    local_k,
                    local_v,
                    out[..., i * head_step : (i + 1) * head_step, :],
                    softmax_lse[..., i * head_step : (i + 1) * head_step, :],
                )
            else:
                dq, dk, dv = _head_other_window_backward(
                    dout[..., i * head_step : (i + 1) * head_step, :],
                    q[..., i * head_step : (i + 1) * head_step, :],
                    local_k,
                    local_v,
                    dq,
                    dk,
                    dv,
                    out[..., i * head_step : (i + 1) * head_step, :],
                    softmax_lse[..., i * head_step : (i + 1) * head_step, :],
                    window_num_idx=j,
                )

            next_dk: torch.Tensor = dkv_comm.send_recv(dk.contiguous())
            next_dv: torch.Tensor = dkv_comm.send_recv(dv.contiguous())
            dkv_comm.commit()

        dkv_comm.wait()
        dqs.append(dq)
        next_dks.append(next_dk)
        next_dvs.append(next_dv)

    dq = torch.cat(dqs, dim=-2).to(q.dtype)
    next_dk = torch.cat(next_dks, dim=-2).to(q.dtype)
    next_dv = torch.cat(next_dvs, dim=-2).to(q.dtype)

    return dq, next_dk, next_dv


def zigzag_ring_flash_attn_backward(
    ring_pg,
    p2p_pg,
    all_gather_pg,
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
        print("P2P + AllGATHER BACKWARD.........", flush=True)
    assert causal is True, "zigzag ring is meaningless for causal=False"

    all_gather_comm = RingComm(all_gather_pg)
    p2p_comm = RingComm(p2p_pg)
    ring_comm = RingComm(ring_pg)
    dkv_comm = RingComm(p2p_pg)

    head_overlap_enable = gpc.config.ring_attn_overlap.get("enable", False)
    if head_overlap_enable:
        head_chunks = gpc.config.ring_attn_overlap.get("head_chunks", 1)
        assert head_chunks > 1, "when enables the head overlap, the head chunks should be > 1."
        assert k.shape[-2] % head_chunks == 0, "the number of head should be divided by the head chunks."
    else:
        head_chunks = 1

    head_step = q.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = create_buffer(q, head_chunks=head_chunks, dim=-2)
    dk_buffer = create_buffer(k, head_chunks=head_chunks, dim=-2)
    dv_buffer = create_buffer(v, head_chunks=head_chunks, dim=-2)

    def get_next_kv_idx(prev_idx) -> int:
        if prev_idx == 0:
            return all_gather_comm.world_size - 1
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

    def _head_first_window_backward(head_dout, head_q, head_k, head_v, head_out, head_softmax_lse):

        full_dk = torch.zeros_like(head_k, dtype=torch.float32)
        full_dv = torch.zeros_like(head_v, dtype=torch.float32)
        dq, dk, dv = None, None, None

        for step in range(all_gather_comm.world_size):

            if step == 0:
                cur_kv_idx = all_gather_comm.rank
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
                if step <= all_gather_comm.rank:
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

                if step <= all_gather_comm.rank:
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

        return dq, full_dk, full_dv

    def _head_other_window_backward(head_dout, head_q, head_k, head_v, head_out, dq, head_softmax_lse, window_num_idx):

        full_dk = torch.zeros_like(head_k, dtype=torch.float32)
        full_dv = torch.zeros_like(head_v, dtype=torch.float32)

        if window_num_idx > p2p_comm.rank:

            for step in range(all_gather_comm.world_size):

                if step == 0:
                    cur_kv_idx = all_gather_comm.rank
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
                full_dk[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len] += dk_buffer.to(
                    torch.float32
                )
                full_dv[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len] += dv_buffer.to(
                    torch.float32
                )
        else:

            for step in range(all_gather_comm.world_size):
                if step == 0:
                    cur_kv_idx = all_gather_comm.rank
                else:
                    cur_kv_idx = get_next_kv_idx(cur_kv_idx)

                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k0 = head_k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                v0 = head_v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                backward(head_dout, head_q, k0, v0, head_out, head_softmax_lse, causal=False)
                dq += dq_buffer

                full_dk[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len] += dk_buffer[
                    :, :block_seq_len
                ].to(torch.float32)

                full_dv[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len] += dv_buffer[
                    :, :block_seq_len
                ].to(torch.float32)

        return dq, full_dk, full_dv

    window_size = gpc.config.ring_attn_overlap.get("window_size", 1)
    window_num = ring_comm.world_size // window_size

    dqs, dks, dvs = [], [], []

    kv_comm_stream = torch.cuda.Stream()
    kv_comm_event = torch.cuda.Event()
    kv_compute_event = torch.cuda.Event()

    dkv_comm_stream = torch.cuda.Stream()
    dkv_comm_event = torch.cuda.Event()
    dkv_compute_event = torch.cuda.Event()

    for i in range(head_chunks):

        local_k = k_splits[i]
        local_v = v_splits[i]

        for j in range(window_num):

            if j > 0:
                torch.cuda.current_stream().wait_event(kv_comm_event)
                full_k = full_k_next
                full_v = full_v_next
                local_k = next_k
                local_v = next_v
            else:
                full_k, _ = all_gather_raw(local_k, all_gather_pg, async_op=False, gather_dim=1)
                full_v, _ = all_gather_raw(local_v, all_gather_pg, async_op=False, gather_dim=1)
                torch.cuda.current_stream().record_event(kv_compute_event)

            if j + 1 != window_num:
                with torch.cuda.stream(kv_comm_stream):
                    kv_comm_stream.wait_event(kv_compute_event)
                    next_k: torch.Tensor = p2p_comm.send_recv(local_k.contiguous())
                    next_v: torch.Tensor = p2p_comm.send_recv(local_v.contiguous())
                    p2p_comm.commit()
                    p2p_comm.wait()
                    full_k_next, _ = all_gather_raw(next_k, all_gather_pg, async_op=False, gather_dim=1)
                    full_v_next, _ = all_gather_raw(next_v, all_gather_pg, async_op=False, gather_dim=1)
                    kv_comm_stream.record_event(kv_comm_event)

            if j == 0:
                dq, full_dk, full_dv = _head_first_window_backward(
                    dout[..., i * head_step : (i + 1) * head_step, :],
                    q[..., i * head_step : (i + 1) * head_step, :],
                    full_k,
                    full_v,
                    out[..., i * head_step : (i + 1) * head_step, :],
                    softmax_lse[:, i * head_step : (i + 1) * head_step, :],
                )
            else:
                dq, full_dk, full_dv = _head_other_window_backward(
                    dout[..., i * head_step : (i + 1) * head_step, :],
                    q[..., i * head_step : (i + 1) * head_step, :],
                    full_k,
                    full_v,
                    out[..., i * head_step : (i + 1) * head_step, :],
                    dq,
                    softmax_lse[:, i * head_step : (i + 1) * head_step, :],
                    window_num_idx=j,
                )

                torch.cuda.current_stream().wait_event(dkv_comm_event)
                dk = next_dk
                dv = next_dv
                full_dk[
                    :, 2 * all_gather_comm.rank * block_seq_len : 2 * (all_gather_comm.rank + 1) * block_seq_len
                ] += dk.to(torch.float32)
                full_dv[
                    :, 2 * all_gather_comm.rank * block_seq_len : 2 * (all_gather_comm.rank + 1) * block_seq_len
                ] += dv.to(torch.float32)

            torch.cuda.current_stream().record_event(kv_compute_event)
            torch.cuda.current_stream().record_event(dkv_compute_event)

            with torch.cuda.stream(dkv_comm_stream):
                dkv_comm_stream.wait_event(dkv_compute_event)
                # reduce-scatter dk and kv
                dk, _ = reduce_scatter_raw(full_dk, all_gather_pg, async_op=False, reduce_dim=1)
                dv, _ = reduce_scatter_raw(full_dv, all_gather_pg, async_op=False, reduce_dim=1)

                next_dk: torch.Tensor = dkv_comm.send_recv(dk.contiguous())
                next_dv: torch.Tensor = dkv_comm.send_recv(dv.contiguous())
                dkv_comm.commit()
                dkv_comm.wait()
                dkv_comm_stream.record_event(dkv_comm_event)

        torch.cuda.current_stream().wait_event(dkv_comm_event)
        dqs.append(dq)
        dks.append(next_dk)
        dvs.append(next_dv)

    dq_final = torch.concat(dqs, dim=-2)
    dk_final = torch.concat(dks, dim=-2)

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
        ring_group,
        p2p_group=None,
        all_gather_group=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()

        sliding_window_comm = gpc.config.ring_attn_overlap.get("comm", "p2p_AG")
        forward_func = (
            zigzag_ring_flash_attn_forward if sliding_window_comm == "p2p_AG" else zigzag_double_ring_flash_attn_forward
        )

        out, softmax_lse = forward_func(
            ring_group,
            p2p_group,
            all_gather_group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.ring_group = ring_group
        ctx.p2p_group = p2p_group
        ctx.all_gather_group = all_gather_group
        ctx.sliding_window_comm = sliding_window_comm
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):  # pylint: disable=W0613
        q, k, v, out, softmax_lse = ctx.saved_tensors

        sliding_window_comm = ctx.sliding_window_comm

        backward_func = (
            zigzag_ring_flash_attn_backward
            if sliding_window_comm == "p2p_AG"
            else zigzag_double_ring_flash_attn_backward
        )

        dq, dk, dv = backward_func(
            ctx.ring_group,
            ctx.p2p_group,
            ctx.all_gather_group,
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
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


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


def zigzag_ring_flash_attn_kvpacked_func_with_sliding_window(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    ring_group=None,
    p2p_group=None,
    all_gather_group=None,
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
        ring_group,
        p2p_group,
        all_gather_group,
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
