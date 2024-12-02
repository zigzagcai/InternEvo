# Adapted from https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py

from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


__all__ = ["update_out_and_lse", "RingComm"]


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


class RingComm:
    """
    P2P communicator for double ring zigzag flash attn.
    """

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self._funcs = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None
        self._handles = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)
            self.global_rank = dist.get_rank()
            # print(f'rank:{self.rank},send_rank:{self.send_rank},recv_rank:{self.recv_rank}')

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.send_rank == self.recv_rank == self.global_rank:
            return to_send

        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        # send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        # recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        # self._ops.append(send_op)
        # self._ops.append(recv_op)

        send_func = partial(dist.isend, tensor=to_send, dst=self.send_rank, group=self._process_group)
        recv_func = partial(dist.irecv, tensor=res, src=self.recv_rank, group=self._process_group)

        if self.rank % 2 == 0:
            self._funcs.append(send_func)
            self._funcs.append(recv_func)
        else:
            self._funcs.append(recv_func)
            self._funcs.append(send_func)

        return res

    def commit(self):
        # if self._reqs is not None:
        #     raise RuntimeError("commit called twice")
        # self._reqs = dist.batch_isend_irecv(self._ops)

        if self._handles is not None:
            raise RuntimeError("commit called twice")
        self._handles = []

        for _func in self._funcs:
            _handle = _func()
            self._handles.append(_handle)

    def wait(self):
        # if self._reqs is None:
        #     raise RuntimeError("wait called before commit")
        # for req in self._reqs:
        #     req.wait()
        # self._reqs = None
        # self._ops = []

        if self._handles is None:
            raise RuntimeError("wait called before commit")
        for _handle in self._handles:
            _handle.wait()
        self._handles = None
        self._funcs = []
