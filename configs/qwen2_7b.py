# Copyright (c) InternLM. All rights reserved.
model_type = "QWEN2"

VOCAB_SIZE = 152064

HIDDEN_SIZE = 3584
NUM_ATTENTION_HEAD = 28
NUM_KV_ATTENTION_HEAD = 4
MLP_RATIO = 1
NUM_LAYER = 28
MULTIPLE_OF = 256

model = dict(
    checkpoint=False,
    num_chunks=1,
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    qkv_bias=True,
    o_bias=False,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-6,
    rope_base=1000000,
    sliding_window_cfg=dict(
        use_sliding_window=False,
        sliding_window=131072,
        max_window_layers=28,
    ),
    multiple_of=MULTIPLE_OF,
    intermediate_size=18944,
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

parallel = dict(
    zero1=dict(size=-1, fsdp=False),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=False, memory_pool=False),
)


JOB_NAME = "qwen2"
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.028
OPTIMIZER_WARMUP_STEP = 0

MICRO_NUM = 4
MICRO_BSZ = 1
SEQ_LEN = 4096
TOTAL_STEP = 75000
PACK_SAMPLE_INTO_ONE = False
USE_PACKED_DATASET = True
SAVED_DATA_PATH = ""

SAVE_CKPT_FOLDER = None
LOAD_MODEL_PATH = None
CHECKPOINT_EVERY = 1000

data = dict(
    seq_len=SEQ_LEN,
    micro_num=MICRO_NUM,
    micro_bsz=MICRO_BSZ,
    valid_micro_num=4,
    valid_every=0,
    pack_sample_into_one=PACK_SAMPLE_INTO_ONE,
    total_steps=TOTAL_STEP,
    skip_batches="",
    rampup_batch_size="",
    min_length=50,
    train_folder=None,
    valid_folder=None,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
    use_packed_dataset=USE_PACKED_DATASET,
)
loss = dict(label_smoothing=0.0)
adam = dict(
    lr=LEARNING_RATE,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=WEIGHT_DECAY,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=OPTIMIZER_WARMUP_STEP,  # optimizer_warmup_step
    warmup_ratio=WARMUP_RATIO,
    eta_min=MIN_LEARNING_RATE,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)
cudnn_deterministic = False
cudnn_benchmark = False
monitor = dict(
    alert=dict(
        enable_feishu_alert=False,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)
grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**14,
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
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=False,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=CHECKPOINT_EVERY,  # snapshot ckpt save frequency.
)
