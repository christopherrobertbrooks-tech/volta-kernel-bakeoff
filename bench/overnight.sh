#!/usr/bin/env bash
# Overnight bake-off (2026-10-19, Chris: "let's do the bake off as the overnight run"), V100 alone:
#  1. ./bakeoff --check           must pass (every contender = a CPU double reference of the same weights)
#  2. BAKEOFF_SABOTAGE=1 --check  must FAIL (the check can catch a wrong kernel)
#  3. the sweep, ROUNDS rounds (default 3) GAP minutes apart (default 0) -> results/<date>/sweep-<n>.csv,
#     with the V100's temperature and SM clock logged before each round (gpu.log)
#  4. llama.cpp's own kernels at the same shapes (bench/llamacpp_mm.cpp), LLAMA_ROUNDS rounds (default 10) -> llama-<n>.csv
#  The SM clock is locked at 1,380 MHz for the run (nvidia-smi -lgc) and reset at the end (-rgc).
#  While each round runs, the card is sampled every 100 ms (gpu-load-<round>.csv: SM/memory clock, power,
#  temperature, utilization, clock-event reasons, volatile uncorrected ECC). bench/load_flag.py flags the round in
#  run.log when a busy sample leaves 1380 MHz or shows a clock-event reason other than idle / the clock setting,
#  when uncorrected ECC is nonzero or rises, or when the file has no usable busy sample (Grok, issue #7).
#  A watchdog logs any other process on the V100 every 10 s (watch.log); summary.py marks rounds it overlapped.
cd "$(dirname "$0")/.." || exit 1
D=${OUT:-results/$(date +%F)}; mkdir -p $D; L=$D/run.log   # OUT=dir to write elsewhere (tests)
say() { echo "$(date '+%F %T') $*" | tee -a $L; }
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
[ -n "$V100" ] || { echo "no V100 found"; exit 1; }
others() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$V100" | grep -v " $1," ; }
[ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$V100")" ] && { say "refusing: the V100 is busy"; exit 1; }
make -s bakeoff >> $L 2>&1 || { say "build FAILED"; exit 1; }
LC=${LLAMA_CPP:-$HOME/llama.cpp-mainline}
g++ -O2 -std=c++17 -I$LC/ggml/include bench/llamacpp_mm.cpp -L$LC/build-sm70/bin -lggml -lggml-base -lggml-cuda \
  -Wl,-rpath,$LC/build-sm70/bin -o llamacpp_mm >> $L 2>&1 || { say "llamacpp_mm build FAILED"; exit 1; }
( while sleep 10; do o=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$V100" | grep -v -E "bakeoff|llamacpp_mm"); [ -n "$o" ] && echo "$(date '+%F %T') OTHER ON V100: $o" >> $D/watch.log; done ) & W=$!
# Lock the V100's SM clock at its maximum, 1,380 MHz, for the whole run and give it back at the end (Chris 2026-10-20,
# after Grok's review: one start-up sample had read 1,230). A power or heat limit can still pull it down; gpu.log shows it.
# The card is named by its UUID everywhere (lock, reset, logging, CUDA_VISIBLE_DEVICES), the same ID the watchdog uses.
trap 'kill $W 2>/dev/null; sudo -n nvidia-smi -i $V100 -rgc >> $L 2>&1' EXIT
sudo -n nvidia-smi -i $V100 -lgc 1380,1380 >> $L 2>&1 || { say "could not lock the V100's clock -- stopping"; exit 1; }
[ "$(nvidia-smi -i $V100 --query-gpu=clocks.sm --format=csv,noheader,nounits)" = 1380 ] || { say "clock lock did not take -- stopping"; exit 1; }
say "V100 SM clock locked at 1380 MHz"
export CUDA_VISIBLE_DEVICES=$V100
say "check"; ./bakeoff --check > $D/check.csv 2>> $L && say "check PASSED ($(grep -c ',ok$' $D/check.csv) ok)" || { say "check FAILED: $(grep -c ',FAIL$' $D/check.csv) failures -- stopping"; exit 1; }
say "sabotage check"; BAKEOFF_SABOTAGE=1 ./bakeoff --check > $D/check-sabotage.csv 2>> $L && { say "SABOTAGE NOT CAUGHT -- the check is broken, stopping"; exit 1; } || say "sabotage caught ($(grep -c ',FAIL$' $D/check-sabotage.csv) failures, as it should)"
# sample the V100 every 100 ms while a round runs. load_flag.py then flags the round if a busy sample left 1380 MHz
# or showed a clock-event reason other than idle (0x1) or the locked clock (0x2), if volatile uncorrected ECC was
# nonzero or rose, or if the csv has no usable busy sample (empty, unparsed, or utilization never above 0).
sample_start() { nvidia-smi -i $V100 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active,ecc.errors.uncorrected.volatile.total --format=csv,noheader -lms 100 > $D/gpu-load-$1.csv 2>/dev/null & S=$!; }
sample_stop() {
  kill $S 2>/dev/null; wait $S 2>/dev/null
  python3 bench/load_flag.py "$1" $D/gpu-load-$1.csv | tee -a $L >> $D/gpu.log
}
for r in $(seq 1 ${ROUNDS:-3}); do
  [ "$r" -gt 1 ] && [ "${GAP:-0}" -gt 0 ] && sleep $(( GAP * 60 ))
  echo "$(date '+%F %T') round $r $(nvidia-smi -i $V100 --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader)" >> $D/gpu.log
  say "sweep round $r"; sample_start $r
  ./bakeoff > $D/sweep-$r.csv 2>> $L && say "round $r done ($(($(wc -l < $D/sweep-$r.csv) - 1)) rows)" || say "round $r FAILED"
  sample_stop $r
done
for r in $(seq 1 ${LLAMA_ROUNDS:-10}); do
  say "llama.cpp round $r"; sample_start llama-$r
  ./llamacpp_mm > $D/llama-$r.csv 2>> $L && say "llama.cpp round $r done ($(($(wc -l < $D/llama-$r.csv) - 1)) rows)" || say "llama.cpp round $r FAILED"
  sample_stop llama-$r
done
say "BAKEOFF DONE; V100 $(nvidia-smi -i $V100 --query-gpu=memory.used --format=csv,noheader); other users of the V100 seen: $(cat $D/watch.log 2>/dev/null | wc -l)"
