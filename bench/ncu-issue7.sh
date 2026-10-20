#!/usr/bin/env bash
# Issue #7, part 2 (after the under-load session): Nsight Compute of tc_q4 and the cuBLAS fp16 GEMM at M=512 on
# qwen36_attn_qkv and qwen36_expert_gate_up (full set, SM clock locked at 1380 and --clock-control none so ncu keeps
# it), and the cuBLAS kernel picked at flashnext_attn_q M=64 vs 128 and qwen3_1p7b_mlp_up M=128 vs 256.
cd /mnt/steam/volta-kernel-bakeoff || exit 1
R=results/$(date +%F)-ncu-tc_q4; mkdir -p $R; L=$R/run.log
say(){ echo "$(date '+%F %T') $*" | tee -a $L; }
until grep -q "BAKEOFF DONE" results/2026-10-20-load/run.log 2>/dev/null; do sleep 30; done
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
[ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$V100")" ] && { say "refusing: V100 busy"; exit 1; }
trap 'sudo -n nvidia-smi -i $V100 -rgc >> $L 2>&1' EXIT
sudo -n nvidia-smi -i $V100 -lgc 1380,1380 >> $L 2>&1 || { say "could not lock the clock"; exit 1; }
NCU="sudo -n env CUDA_VISIBLE_DEVICES=$V100 /usr/local/cuda/bin/ncu --clock-control none"
for S in qwen36_attn_qkv qwen36_expert_gate_up; do
  $NCU --set full -k regex:tc_q4 --launch-skip 5 --launch-count 1 -f -o $R/tc_q4-$S \
    env BAKEOFF_SHAPE=$S BAKEOFF_M=512 ./bakeoff > $R/tc_q4-$S.log 2>&1; say "tc_q4 $S exit $?"
  $NCU --set full -k regex:"gemm|Kernel2|cutlass" --launch-skip 5 --launch-count 1 -f -o $R/cublas-$S \
    env BAKEOFF_SHAPE=$S BAKEOFF_M=512 ./bakeoff > $R/cublas-$S.log 2>&1; say "cublas $S exit $?"
done
for SM in flashnext_attn_q:64 flashnext_attn_q:128 qwen3_1p7b_mlp_up:128 qwen3_1p7b_mlp_up:256; do
  S=${SM%%:*}; M=${SM#*:}
  $NCU --section LaunchStats --section SpeedOfLight -k regex:"gemm|Kernel2|cutlass" --launch-skip 5 --launch-count 1 -f -o $R/cublas-algo-$S-M$M \
    env BAKEOFF_SHAPE=$S BAKEOFF_M=$M ./bakeoff > $R/cublas-algo-$S-M$M.log 2>&1; say "cublas algo $S M=$M exit $?"
done
sudo -n chown -R chris:chris $R
for f in $R/*.ncu-rep; do /usr/local/cuda/bin/ncu --import $f --csv --page details > ${f%.ncu-rep}.csv 2>/dev/null; done
say "NCU DONE ($(ls $R/*.ncu-rep | wc -l) reports)"
