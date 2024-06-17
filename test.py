import numpy

# origin_tgs = [132.909,
# 145.002,
# 151.454,
# 153.614,
# 151.998,
# 145.282]
# origin_mfu = [0.7232,
# 0.789,
# 0.8242,
# 0.8359,
# 0.8271,
# 0.7906]

# tgs = [169.51,
# 182.094,
# 188.326,
# 191.75,
# 189.49,
# 183.252]

# for i in range(len(origin_tgs)):
#     mfu = origin_mfu[i] * tgs[i] / origin_tgs[i]
#     print(mfu)


def get_tflops(seq_len, mlp_ratio=3.5, global_batch_size=1, hidden_size=4096, num_layers=32, vocab_size=103168, global_world_size=64):
    flops_per_iteration = (
        # wqkv wo mlp
        (3 * ((8 + mlp_ratio * 4) * global_batch_size * seq_len * hidden_size**2))
        * num_layers
        # attn
        + 3 * (4 * global_batch_size * seq_len**2 * hidden_size) * num_layers / 2
        # head
        + 6 * global_batch_size * seq_len * hidden_size * vocab_size
    )
        
    flops = flops_per_iteration / (global_world_size * (10**12))
    
    return flops

time = [38.0439326,
33.1351373,
30.7459034,
29.8670711,
30.0557675,
31.5207856]

for t in time:
    mfu = get_tflops(512 * 1024) / t / 312
    print(mfu)

# for seq_len in [128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]:
#     print(get_tflops(seq_len))

# time_cost = num_tokens_in_batch / 64 / tgs_origin
            




    