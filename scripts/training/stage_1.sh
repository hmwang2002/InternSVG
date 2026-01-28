set -ex
cd ./LLaMA-Factory
source /root/miniconda3/etc/profile.d/conda.sh
conda activate internsvg

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "NNODES: $NODE_COUNT"

FORCE_TORCHRUN=1 \
NNODES=${NODE_COUNT} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=29500 \
NPROC_PER_NODE=${PROC_PER_NODE} \
NCCL_SOCKET_IFNAME=bond0 \
llamafactory-cli train ./LLaMA-Factory/examples/train_full/stage_1.yaml
