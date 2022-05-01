#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg configs/swin_base_patch4_window7_224.yaml --data-path ../BIRDDOG --batch-size 128 