#!/usr/bin/env bash
# Issue #3: A/B of a llama.cpp fusion in the private build (~/llama.cpp-fuse, branch fuse; never sent upstream).
# Same binary both ways, the fusion switched off by an environment variable, so nothing else differs. SM clock
# locked at 1380 MHz; ROUNDS rounds of llama-bench decode (alternating which side goes first); then greedy text and
# perplexity both ways -- a fusion that keeps llama.cpp's arithmetic order must give identical text and perplexity.
# Perplexity runs twice per side: llama.cpp's default 512-row batches, and 8-row batches (-ub 8) -- the 1-8 row path
# that the conversion folding uses. The V100 is sampled every 100 ms during each bench run and load_flag.py flags clock
# dips; the log records the llama.cpp commit and a hash of any uncommitted changes, i.e. exactly what was tested.
#   NAME=add-norm OFF_ENV=GGML_CUDA_NO_ADD_NORM_FUSION=1 bench/fuse-ab.sh
#   OFF_ENV may hold several variables: OFF_ENV="GGML_CUDA_NO_ADD_NORM_FUSION=1 GGML_CUDA_NO_Q8_CACHE=1"
cd /mnt/steam/volta-kernel-bakeoff || exit 1
M=${MODEL:-/mnt/steam/Strata-data-qwen36/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
B=${LLAMA_BIN:-$HOME/llama.cpp-fuse/build-sm70/bin}
OFF=${OFF_ENV:?set OFF_ENV to the variable that switches the fusion off}
R=results/$(date +%F)-fuse-${NAME:?set NAME}; mkdir -p $R; L=$R/run.log
say(){ echo "$(date '+%F %T') $*" | tee -a $L; }
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
[ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$V100")" ] && { say "refusing: V100 busy"; exit 1; }
trap 'sudo -n nvidia-smi -i $V100 -rgc >> $L 2>&1' EXIT
sudo -n nvidia-smi -i $V100 -lgc 1380,1380 >> $L 2>&1 || { say "could not lock the clock"; exit 1; }
export CUDA_VISIBLE_DEVICES=$V100
SRC=$(cd "$B/../.." && pwd)
say "model $M; binary $B; off = $OFF"
U=$(git -C $SRC diff --stat | tail -1 | sed 's/^ *//')
say "llama.cpp: $(git -C $SRC log -1 --format='%h %s'); uncommitted: ${U:-none}; diff sha256 $(git -C $SRC diff | sha256sum | cut -c1-16)"
git -C $SRC diff > $R/uncommitted.diff
# rebuild from exactly that tree, so the binary is what the log names (llama-bench's own build_commit can be stale)
cmake --build $B/.. -j 16 --target llama-bench llama-perplexity llama-completion >> $R/build.log 2>&1 || { say "build FAILED"; exit 1; }
say "binaries: $(cd $B && sha256sum libggml-cuda.so* llama-bench llama-perplexity 2>/dev/null | awk '!/->/ {printf "%s %s; ", substr($1,1,16), $2}')"
sample_start() { nvidia-smi -i $V100 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active,ecc.errors.uncorrected.volatile.total --format=csv,noheader -lms 100 > $R/gpu-load-$1.csv 2>/dev/null & S=$!; }
sample_stop() { kill $S 2>/dev/null; wait $S 2>/dev/null; python3 bench/load_flag.py "$1" $R/gpu-load-$1.csv >> $L; }
for r in $(seq 1 ${ROUNDS:-10}); do
  for side in $( ((r % 2)) && echo "on off" || echo "off on"); do
    if [ $side = off ]; then E="env $OFF"; else E=env; fi
    sample_start $side-$r
    $E $B/llama-bench -m $M -ngl 99 -fa 1 -p 0 -n 128 -r 3 -o csv > $R/bench-$side-$r.csv 2>> $R/bench.err
    sample_stop $side-$r
    say "round $r $side: $(tail -1 $R/bench-$side-$r.csv | awk -F, '{print $(NF-1)}' | tr -d '\"') tok/s"
  done
done
P="The history of the printing press begins in the fifteenth century, when"
for side in on off; do
  if [ $side = off ]; then E="env $OFF"; else E=env; fi
  $E $B/llama-completion -m $M -ngl 99 -fa 1 -p "$P" -n 256 --temp 0 --top-k 1 --seed 1 --ignore-eos -no-cnv \
      > $R/text-$side.txt 2> $R/text-$side.err
  for ub in 512 8; do
    $E $B/llama-perplexity -m $M -ngl 99 -fa 1 -f /mnt/steam/gsq-gguf/results/ppl/wiki.test.raw -c 512 --chunks 40 \
        -ub $ub > $R/ppl-$side-ub$ub.txt 2>&1
    grep -o '\[[0-9]*\][0-9.]*' $R/ppl-$side-ub$ub.txt > $R/ppl-chunks-$side-ub$ub.txt
  done
  say "$side: text $(wc -c < $R/text-$side.txt) bytes; ub 512 $(grep -o 'Final estimate.*' $R/ppl-$side-ub512.txt); ub 8 $(grep -o 'Final estimate.*' $R/ppl-$side-ub8.txt)"
done
cmp -s $R/text-on.txt $R/text-off.txt && say "greedy text IDENTICAL" || say "greedy text DIFFERS"
for ub in 512 8; do
  n=$(wc -l < $R/ppl-chunks-on-ub$ub.txt)
  cmp -s $R/ppl-chunks-on-ub$ub.txt $R/ppl-chunks-off-ub$ub.txt && say "perplexity -ub $ub: all $n chunks IDENTICAL" || say "perplexity -ub $ub: chunks DIFFER"
done
# logit-level identity: KL divergence of the fused side against the stock side's saved log-probabilities (4 chunks;
# identical output gives Mean KLD 0 and same top token 100%). The base files (~1 GB each) go to /tmp and to the trash.
for ub in 512 8; do
  KB=/tmp/fuse-ab-kld-$NAME-ub$ub.bin
  env $OFF $B/llama-perplexity -m $M -ngl 99 -fa 1 -f /mnt/steam/gsq-gguf/results/ppl/wiki.test.raw -c 512 --chunks 4 \
      -ub $ub --kl-divergence-base $KB > $R/kld-base-ub$ub.txt 2>&1
  $B/llama-perplexity -m $M -ngl 99 -fa 1 -c 512 --chunks 4 -ub $ub --kl-divergence-base $KB --kl-divergence \
      > $R/kld-on-ub$ub.txt 2>&1
  say "KL divergence -ub $ub: $(grep -E 'Mean +KLD|Maximum KLD|Same top p|RMS Δp' $R/kld-on-ub$ub.txt | tr -s ' ' | tr '\n' ';')"
  gio trash $KB 2>/dev/null
done
say "clock: $(grep -c FLAGGED $L) of $(grep -c 'under load' $L) bench runs flagged; $(grep -o '[0-9]* below 1380' $L | awk '{s+=$1} END {print s+0}') busy samples below 1380 MHz"
say "FUSE-AB DONE"
