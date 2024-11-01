#! /bin/bash

export PYTHONPATH=/share/work/sh/work/20241016/third/transformers/src:$PYTHONPATH

# Setting the environment variables
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PATH=/opt/maca/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:${PATH}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export SET_DEVICE_NUMA_PREFERRED=1
export MAX_JOBS=20
export MCCL_RING_ACCURACY=1
export FORCE_ACTIVE_WAIT=2
export MCCL_FAST_WRITE_BACK=1
export MCCL_EARLY_WRITE_BACK=15
export GLOO_SOCKET_IFNAME=manage0
export MCCL_ENABLE_FC=0
# export MCCL_MAX_NCHANNELS=18
export MCCL_ALGO=Ring
export MCCL_P2P_LEVEL=SYS
export MCCL_NET_GDR_LEVEL=SYS
export MCCL_CROSS_NIC=1
# export vmmDefragment=1
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export MCCL_IB_HCA=mlx5_0,mlx5_1

export PYTORCH_ENABLE_SAME_RAND_A100=1

NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
HOST_NAME=`hostname`
NODE_RANK=0
MASTER_PORT=12345
MASTER_ADDR="10.77.2.6"
echo NNODES=$NNODES NODE_RANK=$NODE_RANK HOST_FILE=${HOST_FILE} HOST_NAME=${HOST_NAME} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}

LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
       /share/work/sh/work/20241016/InternEvo/train.py \
         --config /share/work/sh/work/20241016/InternEvo/configs/ZHH_7B.py --launcher "torch"
       "

${LAUNCHER}


