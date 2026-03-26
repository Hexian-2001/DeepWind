#!/bin/bash

GPUS=8

python -m torch.distributed.run --nproc_per_node=$GPUS --standalone evaluate.py
    

