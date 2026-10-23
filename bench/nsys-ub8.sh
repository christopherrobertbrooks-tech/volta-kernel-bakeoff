#!/usr/bin/env bash
# Issue #3 (Grok's review): show that the norm folding runs in llama-perplexity with 8-row batches. Nsight Systems traces
# `llama-perplexity -ub 8 --chunks 2` with everything on and with everything off (stock), and counts per graph run:
# the fold kernel (add_rms_norm_mul_f32<1024, 0>, plain norm -> mul), all add_rms_norm_mul_f32, and quantize_q8_1.
#   bench/nsys-ub8.sh OUT_DIR
cd /mnt/steam/volta-kernel-bakeoff || exit 1
R=${1:?OUT_DIR}; mkdir -p $R
M=${MODEL:-/mnt/steam/Strata-data-qwen36/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
B=${LLAMA_BIN:-$HOME/llama.cpp-fuse/build-sm70/bin}
SRC=$(cd "$B/../.." && pwd)
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
[ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$V100")" ] && { echo "refusing: V100 busy"; exit 1; }
OFF="GGML_CUDA_NO_ADD_NORM_FUSION=1 GGML_CUDA_NO_Q8_CACHE=1 GGML_CUDA_NO_MMVF_SIGMOID_FUSION=1"
{ echo "llama.cpp: $(git -C $SRC log -1 --format='%h %s'); diff sha256 $(git -C $SRC diff | sha256sum | cut -c1-16)"
  echo "libggml-cuda sha256 $(sha256sum $B/libggml-cuda.so.0.* | cut -c1-16)"
  echo "off = $OFF"
  echo "per side: CUDA_VISIBLE_DEVICES=<V100> [env \$OFF] nsys profile -o ub8-<side> --force-overwrite true -t cuda \\"
  echo "    $B/llama-perplexity -m $M -ngl 99 -fa 1 -f /mnt/steam/gsq-gguf/results/ppl/wiki.test.raw -c 512 --chunks 2 -ub 8"
  echo "  then: nsys stats --report cuda_gpu_kern_sum --format csv --output ub8-<side> ub8-<side>.nsys-rep"; } > $R/COMMAND.txt
for side in on off; do
  if [ $side = off ]; then E="env $OFF"; else E=env; fi
  CUDA_VISIBLE_DEVICES=$V100 $E nsys profile -o $R/ub8-$side --force-overwrite true -t cuda \
    $B/llama-perplexity -m $M -ngl 99 -fa 1 -f /mnt/steam/gsq-gguf/results/ppl/wiki.test.raw -c 512 --chunks 2 -ub 8 \
    > $R/ub8-$side.log 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv --output $R/ub8-$side $R/ub8-$side.nsys-rep > /dev/null 2>&1
done
python3 - $R <<'EOF'
import csv, sys
r = sys.argv[1]
for side in ("on", "off"):
    rows = list(csv.DictReader(open(f"{r}/ub8-{side}_cuda_gpu_kern_sum.csv")))
    n = lambda pat: sum(int(x["Instances"]) for x in rows if pat in x["Name"])
    fold = sum(int(x["Instances"]) for x in rows if "add_rms_norm_mul_f32<(int)1024, (int)0>" in x["Name"])
    print(f"{side}: kernels {n('')}, quantize_q8_1 {n('quantize_q8_1')}, add_rms_norm_mul_f32 {n('add_rms_norm_mul_f32')} "
          f"(of which the plain-norm fold <1024,0>: {fold}), mul_mat_vec_q {n('mul_mat_vec_q')}")
EOF
