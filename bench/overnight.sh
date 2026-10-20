#!/usr/bin/env bash
# Overnight bake-off (2026-10-19, Chris: "let's do the bake off as the overnight run"), V100 alone:
#  1. ./bakeoff --check           must pass (every contender = a CPU double reference of the same weights)
#  2. BAKEOFF_SABOTAGE=1 --check  must FAIL (the check can catch a wrong kernel)
#  3. the sweep, ROUNDS rounds (default 3) GAP minutes apart (default 0) -> results/<date>/sweep-<n>.csv,
#     with the V100's temperature and SM clock logged before each round (gpu.log)
#  4. llama.cpp's own kernels at the same shapes (bench/llamacpp_mm.cpp), LLAMA_ROUNDS rounds (default 10) -> llama-<n>.csv
#  A watchdog logs any other process on the V100 every 10 s (watch.log); summary.py marks rounds it overlapped.
cd "$(dirname "$0")/.." || exit 1
D=results/$(date +%F); mkdir -p $D; L=$D/run.log
say() { echo "$(date '+%F %T') $*" | tee -a $L; }
V100=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}')
others() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$V100" | grep -v " $1," ; }
[ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$V100")" ] && { say "refusing: the V100 is busy"; exit 1; }
make -s bakeoff >> $L 2>&1 || { say "build FAILED"; exit 1; }
LC=${LLAMA_CPP:-$HOME/llama.cpp-mainline}
g++ -O2 -std=c++17 -I$LC/ggml/include bench/llamacpp_mm.cpp -L$LC/build-sm70/bin -lggml -lggml-base -lggml-cuda \
  -Wl,-rpath,$LC/build-sm70/bin -o llamacpp_mm >> $L 2>&1 || { say "llamacpp_mm build FAILED"; exit 1; }
( while sleep 10; do o=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$V100" | grep -v -E "bakeoff|llamacpp_mm"); [ -n "$o" ] && echo "$(date '+%F %T') OTHER ON V100: $o" >> $D/watch.log; done ) & W=$!
trap 'kill $W 2>/dev/null' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
say "check"; ./bakeoff --check > $D/check.csv 2>> $L && say "check PASSED ($(grep -c ',ok$' $D/check.csv) ok)" || { say "check FAILED: $(grep -c ',FAIL$' $D/check.csv) failures -- stopping"; exit 1; }
say "sabotage check"; BAKEOFF_SABOTAGE=1 ./bakeoff --check > $D/check-sabotage.csv 2>> $L && { say "SABOTAGE NOT CAUGHT -- the check is broken, stopping"; exit 1; } || say "sabotage caught ($(grep -c ',FAIL$' $D/check-sabotage.csv) failures, as it should)"
for r in $(seq 1 ${ROUNDS:-3}); do
  [ "$r" -gt 1 ] && [ "${GAP:-0}" -gt 0 ] && sleep $(( GAP * 60 ))
  echo "$(date '+%F %T') round $r $(nvidia-smi -i 1 --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader)" >> $D/gpu.log
  say "sweep round $r"; ./bakeoff > $D/sweep-$r.csv 2>> $L && say "round $r done ($(($(wc -l < $D/sweep-$r.csv) - 1)) rows)" || say "round $r FAILED"
done
for r in $(seq 1 ${LLAMA_ROUNDS:-10}); do
  say "llama.cpp round $r"; ./llamacpp_mm > $D/llama-$r.csv 2>> $L && say "llama.cpp round $r done ($(($(wc -l < $D/llama-$r.csv) - 1)) rows)" || say "llama.cpp round $r FAILED"
done
say "BAKEOFF DONE; V100 $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader); other users of the V100 seen: $(wc -l < $D/watch.log 2>/dev/null || echo 0)"
