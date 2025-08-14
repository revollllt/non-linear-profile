#!/bin/bash

for seq_len in 1024 2048 4096 8192 16384; do
    for attn_impl in "eager" "triton" "triton_wo_softmax"; do
        echo "Testing: seq_length=$seq_len, attn_implementation=$attn_impl"
        python main.py --seq_length $seq_len --attn_implementation $attn_impl
    done
done