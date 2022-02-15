#!/bin/bash

DIR=$(cd `dirname $0`; pwd)
cd $DIR

output_dir="checkpoint"
mkdir $output_dir

# python3 -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --use_env \
#         ddp_train.py --output-dir $output_dir > $output_dir"/log.txt"

# python3 -m torch.distributed.launch \
#         --nproc_per_node=8 \
#         --nnodes 2 \
#         --node_rank 0 \
#         --master_addr='10.100.37.21' \
#         --master_port='29500' \
#         ddp_train.py --output-dir $output_dir > $output_dir"/log.txt"

python3 single_train.py --output-dir $output_dir > $output_dir"/log.txt"
