#!/bin/bash

export NCCL_DEBUG=info
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

data_path='/path/to/dataset'

# CIFAR10
python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=${port} main_cls.py \
--data_path ${data_path} \
--weight_decay 0.1 \
--momentum_model 0.99 \
--epochs 1000 \
--batch_size 256 \
--learning_rate 2e-4 \
--resume \
--num_of_cat 10\
--data_type cifar10 \

## CIFAR100
#python -m torch.distributed.launch \
#--nproc_per_node=2 --master_port=${port} main_cls.py \
#--data_path ${data_path} \
#--weight_decay 0.1 \
#--momentum_model 0.99 \
#--epochs 1000 \
#--batch_size 256 \
#--learning_rate 2e-4 \
#--resume \
#--num_of_cat 100 \
#--data_type cifar100 \
