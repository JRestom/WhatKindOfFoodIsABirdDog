#!/bin/bash
python main.py  --epochs 10 \
                --model convnext_base \
                --data_set image_folder \
                --data_path ../DOG/train \
                --eval_data_path ../DOG/val \
                --nb_classes 120 \
                --num_workers 1 \
                --warmup_epochs 0 \
                --save_ckpt false \
                --output_dir model_ckpt \
                --finetune convnext_base_1k_224.pth \
                --cutmix 0 \
                --mixup 0 \
                --lr 4e-4 \
                --enable_wandb true \
                --wandb_ckpt true \
                --food false
