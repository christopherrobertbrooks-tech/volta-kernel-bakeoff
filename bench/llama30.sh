#!/usr/bin/env bash
# All of llama.cpp's contenders (incl. Q2_0) for 30 back-to-back rounds in one session, to match ours and cuBLAS (30).
D=/mnt/steam/volta-kernel-bakeoff/results/2026-10-20-llama30; mkdir -p $D; L=$D/run.log
say(){ echo "$(date "+%F %T") $*" >> $L; }
[ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ] && { say "refusing: V100 busy"; exit 1; }
nvidia-smi -i 1 --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv -l 30 > $D/gpu.log & G=$!
export CUDA_VISIBLE_DEVICES=1
for r in $(seq 1 30); do /mnt/steam/volta-kernel-bakeoff/llamacpp_mm > $D/llama-$r.csv 2>> $D/llama.err; say "round $r exit $? ($(($(wc -l < $D/llama-$r.csv) - 1)) rows)"; done
kill $G; say "LLAMA30 DONE"
