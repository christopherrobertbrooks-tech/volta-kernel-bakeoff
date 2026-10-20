#!/usr/bin/env bash
# After the tc_q4_v2 verdict (#1): Nsight Compute of tc_q4_v2 (its default pick) and the cuBLAS fp16 GEMM at M=512 on
# the two worst tables, qwen36_attn_qkv and qwen3_1p7b_mlp_down -- does shared memory still limit it, or the unpack,
# or the global loads? (Full set, SM clock locked at 1380, --clock-control none so ncu keeps it.)
cd /mnt/steam/volta-kernel-bakeoff || exit 1
R=results/$(date +%F)-ncu-tc_q4_v2; mkdir -p $R; L=$R/run.log
say(){ echo "$(date '+%F %T') $*" | tee -a $L; }
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
[ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$V100")" ] && { say "refusing: V100 busy"; exit 1; }
trap 'sudo -n nvidia-smi -i $V100 -rgc >> $L 2>&1' EXIT
sudo -n nvidia-smi -i $V100 -lgc 1380,1380 >> $L 2>&1 || { say "could not lock the clock"; exit 1; }
NCU="sudo -n env CUDA_VISIBLE_DEVICES=$V100 /usr/local/cuda/bin/ncu --clock-control none"
for S in qwen36_attn_qkv qwen3_1p7b_mlp_down; do
  $NCU --set full --import-source yes -k regex:tc_q4_v2 --launch-skip 5 --launch-count 1 -f -o $R/tc_q4_v2-$S \
    env BAKEOFF_SHAPE=$S BAKEOFF_M=512 ./bakeoff > $R/tc_q4_v2-$S.log 2>&1; say "tc_q4_v2 $S exit $?"
  $NCU --set full -k regex:"gemm|Kernel2|cutlass" --launch-skip 5 --launch-count 1 -f -o $R/cublas-$S \
    env BAKEOFF_SHAPE=$S BAKEOFF_M=512 ./bakeoff > $R/cublas-$S.log 2>&1; say "cublas $S exit $?"
done
sudo -n chown -R chris:chris $R
for f in $R/*.ncu-rep; do /usr/local/cuda/bin/ncu --import $f --csv --page details > ${f%.ncu-rep}.csv 2>/dev/null; done
say "NCU DONE ($(ls $R/*.ncu-rep | wc -l) reports)"
