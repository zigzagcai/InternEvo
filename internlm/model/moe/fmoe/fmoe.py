from typing import Callable, Dict, Optional, Tuple

import torch
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.moe.base_layer import BaseMoELayer

from fmoe import FMoE
import fmoe

class FMoELayer(BaseMoELayer, FMoE):
    def __init__(self, in_features: int,
            hidden_features: int,
            out_features: int,
            ep_group,
            num_experts: int = 1,
            ep_size=1,
            top_k: int = 1,
            dtype=None,
            device: Optional[torch.device] = None,
            gate_bias=True,):
        """create a fast-moe layer

        Args:
            ep_group (_type_): _description_
            num_experts (int, optional): global expert count.
            ep_size (int, optional): exprt parallel size.
            top_k (int, optional): gate top-k. Defaults to 1.
        """

        self.num_local_experts = num_experts // ep_size

        # create experts
        experts =  torch.nn.ModuleList(
                [
                    new_feed_forward(
                        in_features,
                        hidden_features,
                        out_features,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(num_experts // ep_size)
                ])

        BaseMoELayer.__init__(self,
            None,
            experts,
            ep_group,
            ep_size,
            self.num_local_experts,
        )
        # breakpoint()
        original_expert = self.experts
        FMoE.__init__(self,
            num_expert=self.num_local_experts,
            d_model=in_features,
            world_size=ep_size,
            mp_group=None,
            top_k=top_k,
            moe_group=ep_group,
            gate = fmoe.gates.GShardGate,
        )
        self.experts = original_expert.wrapped_experts
        self.experts_fused = False
        # breakpoint()

        # gate_cls = fmoe.gates.GShardGate
        # if issubclass(gate_cls, fmoe.gates.NaiveGate):
        #     self.gate = gate_cls(in_features, self.num_local_experts, ep_size, top_k, gate_bias=gate_bias)
        # else:
        #     self.gate = gate_cls(in_features, self.num_local_experts, ep_size, top_k)
        # self.gate_hook = None
        # self.mask = None
        # self.mask_dict = None
        # self.moe_group = ep_group
        # self.world_size=ep_size
        # self.d_model=in_features
        # self.num_expert=self.num_local_experts
        # self.experts = experts
        # self.experts_fused = False

    def forward(self, hidden_states, used_token=None):
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])
        fmoe_out = FMoE.forward(self, hidden_states)
        reshaped_out = fmoe_out.reshape(original_shape[0], -1, original_shape[-1])

        self.l_aux = self.gate.loss

        return reshaped_out

    # override this since exprt fwd do not accept fwd_expert_count
    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        # breakpoint()
        if self.experts_fused:
            # if torch.is_tensor(fwd_expert_count):
            #     fwd_expert_count = fwd_expert_count.tolist()
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_local_experts):
            batch_size = fwd_expert_count_cpu[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            # outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count[i]])))
            outputs.append(self.experts[i](inp_slice))
            # outputs.append(self.experts.forward_single(i, inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)