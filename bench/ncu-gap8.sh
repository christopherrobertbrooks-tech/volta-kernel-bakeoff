#!/usr/bin/env bash
# Issue #2: Nsight Compute of our 8-token 4-bit kernel (fused_q4_v2<8>) and llama.cpp's Q4_0 path at 8 tokens (its
# activation quantize + mul_mat_vec_q) on flashnext_attn_q and qwen36_expert_gate_up. SM clock locked at 1380.
cd /mnt/steam/volta-kernel-bakeoff || exit 1
R=results/$(date +%F)-ncu-gap8; mkdir -p $R; L=$R/run.log
say(){ echo "$(date '+%F %T') $*" | tee -a $L; }
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
[ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$V100")" ] && { say "refusing: V100 busy"; exit 1; }
trap 'sudo -n nvidia-smi -i $V100 -rgc >> $L 2>&1' EXIT
sudo -n nvidia-smi -i $V100 -lgc 1380,1380 >> $L 2>&1 || { say "could not lock the clock"; exit 1; }
NCU="sudo -n env CUDA_VISIBLE_DEVICES=$V100 /usr/local/cuda/bin/ncu --clock-control none"
for S in flashnext_attn_q qwen36_expert_gate_up; do
  $NCU --set full --import-source yes -k regex:"^fused_q4_v2$" --launch-skip 5 --launch-count 1 -f -o $R/ours-$S \
    env BAKEOFF_SHAPE=$S BAKEOFF_M=8 ./bakeoff > $R/ours-$S.log 2>&1; say "ours $S exit $?"
  $NCU --set full -k regex:"mul_mat_vec_q|quantize_q8_1" --launch-skip 10 --launch-count 2 -f -o $R/llama-$S \
    env LLAMA_MM_SHAPE=$S LLAMA_MM_TYPE=llama_q4_0 LLAMA_MM_M=8 ./llamacpp_mm > $R/llama-$S.log 2>&1; say "llama $S exit $?"
done
sudo -n chown -R chris:chris $R
for f in $R/*.ncu-rep; do /usr/local/cuda/bin/ncu --import $f --csv --page raw > ${f%.ncu-rep}-raw.csv 2>/dev/null; /usr/local/cuda/bin/ncu --import $f --csv --page details > ${f%.ncu-rep}.csv 2>/dev/null; done
say "NCU DONE ($(ls $R/*.ncu-rep | wc -l) reports)"
