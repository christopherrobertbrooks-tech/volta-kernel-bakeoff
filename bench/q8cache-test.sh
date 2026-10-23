#!/usr/bin/env bash
# Issue #11: run bench/q8cache_test.cpp against the private llama.cpp with the q8_1 conversion cache on and off, with
# and without llama.cpp's concurrent streams (GGML_CUDA_GRAPH_OPT=1), and require byte-identical outputs. Also checks
# that the instrument can see a stale reuse: with the cache off, run C (input 2) must differ from run B (input 1).
#   bench/q8cache-test.sh [OUT_DIR]     -- OUT_DIR keeps the output files and the log
cd /mnt/steam/volta-kernel-bakeoff || exit 1
LC=${LLAMA_CPP:-$HOME/llama.cpp-fuse}
T=${1:-$(mktemp -d /tmp/q8cache-test.XXXX)}; mkdir -p "$T"
g++ -O2 -std=c++17 -I$LC/ggml/include bench/q8cache_test.cpp -L$LC/build-sm70/bin -lggml -lggml-base -lggml-cuda \
  -Wl,-rpath,$LC/build-sm70/bin -o $T/q8cache_test || { echo "build FAILED"; exit 1; }
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
export CUDA_VISIBLE_DEVICES=$V100
echo "llama.cpp: $(git -C $LC log -1 --format='%h %s'); uncommitted diff sha256 $(git -C $LC diff | sha256sum | cut -c1-16); libggml-cuda sha256 $(sha256sum $LC/build-sm70/bin/libggml-cuda.so.0.* | cut -c1-16)"
fail=0
for opt in 0 1; do
  GGML_CUDA_GRAPH_OPT=$opt $T/q8cache_test $T/on-$opt.bin > /dev/null 2>&1 || { echo "run failed (cache on, GRAPH_OPT=$opt)"; fail=1; }
  GGML_CUDA_GRAPH_OPT=$opt GGML_CUDA_NO_Q8_CACHE=1 $T/q8cache_test $T/off-$opt.bin > /dev/null 2>&1 || { echo "run failed (cache off, GRAPH_OPT=$opt)"; fail=1; }
  python3 bench/q8cache_compare.py $T/on-$opt.bin $T/off-$opt.bin "GRAPH_OPT=$opt" || fail=1
done
[ $fail = 0 ] && echo "Q8CACHE TESTS PASSED" || { echo "Q8CACHE TESTS FAILED"; exit 1; }
