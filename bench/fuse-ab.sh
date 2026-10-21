#!/usr/bin/env bash
# Issue #3: A/B of a llama.cpp fusion in the private build (~/llama.cpp-fuse, branch fuse; never sent upstream).
# Same binary both ways, the fusion switched off by an environment variable, so nothing else differs. SM clock
# locked at 1380 MHz; ROUNDS rounds of llama-bench decode (alternating which side goes first); then greedy text and
# perplexity both ways -- a fusion that keeps llama.cpp's arithmetic order must give identical text and perplexity.
#   NAME=add-norm OFF_ENV=GGML_CUDA_NO_ADD_NORM_FUSION=1 bench/fuse-ab.sh
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
say "model $M; binary $B; off = $OFF; $(cd "$B/../.." && git log -1 --format='%h %s')"
for r in $(seq 1 ${ROUNDS:-10}); do
  for side in $( ((r % 2)) && echo "on off" || echo "off on"); do
    if [ $side = off ]; then E="env $OFF"; else E=env; fi
    $E $B/llama-bench -m $M -ngl 99 -fa 1 -p 0 -n 128 -r 3 -o csv > $R/bench-$side-$r.csv 2>> $R/bench.err
    say "round $r $side: $(tail -1 $R/bench-$side-$r.csv | awk -F, '{print $(NF-1)}' | tr -d '\"') tok/s"
  done
done
P="The history of the printing press begins in the fifteenth century, when"
for side in on off; do
  if [ $side = off ]; then E="env $OFF"; else E=env; fi
  $E $B/llama-completion -m $M -ngl 99 -fa 1 -p "$P" -n 256 --temp 0 --top-k 1 --seed 1 --ignore-eos -no-cnv \
      > $R/text-$side.txt 2> $R/text-$side.err
  $E $B/llama-perplexity -m $M -ngl 99 -fa 1 -f /mnt/steam/gsq-gguf/results/ppl/wiki.test.raw -c 512 --chunks 40 \
      > $R/ppl-$side.txt 2>&1
  say "$side: text $(wc -c < $R/text-$side.txt) bytes; $(grep -o 'Final estimate.*' $R/ppl-$side.txt)"
done
cmp -s $R/text-on.txt $R/text-off.txt && say "greedy text IDENTICAL" || say "greedy text DIFFERS"
say "FUSE-AB DONE"
