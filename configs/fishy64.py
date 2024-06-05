import os

from internlm.utils.utils import read_base

with read_base():
    from configs.mozi import *


MLP_RATIO = 0.5
NUM_EXPERTS = 64
NUM_SELECTED_EXPERTS = 2
model.update(
    {
        "num_experts": NUM_EXPERTS,
        "mlp_ratio": MLP_RATIO,
        # "top_k": NUM_SELECTED_EXPERTS,
        # "moe_type": "MegaBlock-D"
        # "moe_type": "MegaBlock"
        # "moe_type": "FMoe"
        "moe_type": "GShard",
        "num_layers": 8,
    }
)

# GShard MoE config
# moe = dict(
#     top_k=NUM_SELECTED_EXPERTS,
#     capacity_factor=1.2,
#     eval_capacity_factor=1.0,
#     min_capacity=4,
#     noisy_gate_policy=None,
#     drop_tokens=True,
#     use_rts=True,
#     use_fused_gating=False,
# )


# MegaBlock MoE config
moe = dict(
    top_k=NUM_SELECTED_EXPERTS,
    # capacity_factor=1., # only used in MegaBlock(non-dmoe)
    # drop_tokens=True, # only used in MegaBlock(non-dmoe)
    # parallel_mode="tensor", # only used in MegaBlock-D(dmoe), parallel_mode can be tensor or weight
)

JOB_NAME = "fishy64"

expert_parallel_size=8
use_tutel=True

