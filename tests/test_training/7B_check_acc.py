import os

JOB_NAME = "7b_internlm2_train"
model_type = "INTERNLM2"
DO_ALERT = False

VOCAB_SIZE = 92544
SEQ_LEN = 2048
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 3.5
NUM_LAYER = 32

MODEL_ONLY_FOLDER = os.path.join(
    os.environ["share_path"], "quailty_assurance/7B_internlm2_init_dp=2_tp=2_pp=2_ckpt/init"
)
# Ckpt folder format:
# fs: 'local:/mnt/nfs/XXX'
# SAVE_CKPT_FOLDER = "local:llm_ckpts_0925_9"
# LOAD_CKPT_FOLDER = "local:llm_ckpts/49"

# boto3 Ckpt folder format:
# import os
# BOTO3_IP = os.environ["BOTO3_IP"] # boto3 bucket endpoint
# SAVE_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm"
# LOAD_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm/snapshot/1/"
CHECKPOINT_EVERY = 100
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    auto_resume=False,
    # save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    # load_ckpt_folder= dict(path=MODEL_ONLY_FOLDER, content=["model"], ckpt_type="normal"),
    # load_ckpt_folder="local:llm_ckpts/",
    # # 'load_ckpt_info' setting guide:
    # # 1. the 'path' indicate ckpt path,
    # # 2. the 'content‘ means what states will be loaded, support: "model", "sampler", "optimizer", "scheduler", "all"
    # # 3. the ’ckpt_type‘ means the type of checkpoint to be loaded, now only 'normal' type is supported.
    load_ckpt_info=dict(path=MODEL_ONLY_FOLDER, content=("model",), ckpt_type="internevo"),
    # checkpoint_every=CHECKPOINT_EVERY,
    # async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    # async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    # oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

TRAIN_FOLDER = "/path/to/dataset"
VALID_FOLDER = "/path/to/dataset"
data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=2,
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=2,
    # defaults to the value of micro_num
    valid_micro_num=2,
    # defaults to 0, means disable evaluate
    valid_every=5000,
    pack_sample_into_one=False,
    total_steps=501,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    # train_folder=TRAIN_FOLDER,
    # valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=500,
    diag_outlier_ratio=1.1,
    num_worker=4,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

model = dict(
    checkpoint=False,
    num_chunks=1,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    no_bias=True,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    use_flash_attn=True,
    # Whether the odd and even columns of the query and key in the model are normally interleaved.
    # If it's True, the model's odd and even columns are normally ordered; if it's False,
    # it means that the model has prematurely concatenated all odd columns and even columns in front
    # and back, in order to improve the RoPE's computational efficiency.
    # Example:
    # qk_interleaved = True: q[-1] = [q1,q2,q3,q4,q5,q6,...], k[-1] = [k1,k2,k3,k4,k5,k6,...]
    # qk_interleaved = False: q[-1] = [q1,q3,q5,...,q2,q4,q6,...], k[-1] = [k1,k3,k5,...,k2,k4,k6,...]
    qk_interleaved=False,
)
"""
zero1 parallel:
    1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
        so parameters will be divided within the range of dp.
    2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
    3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler.
tensor parallel: tensor parallel size, usually the number of GPUs per node.
"""
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=2, mode="mtp"),
    pipeline=dict(size=2, interleaved_overlap=True),
    weight=dict(size=1, overlap=True),
)

cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)

# metric_dtype can be "fp32" or other string
# only when set to "fp32" will use fp32 to calc in metrics
# metric_dtype = "fp32"

generation = dict(
    ckpt_folder="/path/to/saved/ckpt",
    output_folder="/path/to/save/generation",
    batch_size=1,
    eos_id=[2, 0],
    bos_id=1,
    max_length=100,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1,
    length_penalty=1.0,
)

enable_tb = False
