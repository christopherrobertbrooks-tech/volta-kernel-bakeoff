#!/usr/bin/env bash
# Issue #11: run bench/q8cache_test.cpp against the private llama.cpp with the q8_1 conversion cache on and off, with
# and without llama.cpp's concurrent streams (GGML_CUDA_GRAPH_OPT=1), and require byte-identical outputs. Also checks
# that run C (graph B twice back to back, unsynced, ending on B's input) equals run B.
cd /mnt/steam/volta-kernel-bakeoff || exit 1
LC=${LLAMA_CPP:-$HOME/llama.cpp-fuse}
T=$(mktemp -d /tmp/q8cache-test.XXXX)
g++ -O2 -std=c++17 -I$LC/ggml/include bench/q8cache_test.cpp -L$LC/build-sm70/bin -lggml -lggml-base -lggml-cuda \
  -Wl,-rpath,$LC/build-sm70/bin -o $T/q8cache_test || { echo "build FAILED"; exit 1; }
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/V100/{print $2}')
export CUDA_VISIBLE_DEVICES=$V100
fail=0
for opt in 0 1; do
  GGML_CUDA_GRAPH_OPT=$opt $T/q8cache_test $T/on-$opt.bin > /dev/null 2>&1 || { echo "run failed (cache on, GRAPH_OPT=$opt)"; fail=1; }
  GGML_CUDA_GRAPH_OPT=$opt GGML_CUDA_NO_Q8_CACHE=1 $T/q8cache_test $T/off-$opt.bin > /dev/null 2>&1 || { echo "run failed (cache off, GRAPH_OPT=$opt)"; fail=1; }
  if cmp -s $T/on-$opt.bin $T/off-$opt.bin; then echo "GRAPH_OPT=$opt: cache on == cache off (A in-place, B branches, C back to back)"
  else echo "GRAPH_OPT=$opt: cache on DIFFERS from cache off"; fail=1; fi
  # B is the 3rd block of N floats, C the 4th: C (rerun on B's input) must equal B
  python3 - $T/on-$opt.bin <<'EOF' || fail=1
import sys
d = open(sys.argv[1], "rb").read(); n = 2048 * 4
b, c = d[2 * n:3 * n], d[3 * n:4 * n]
print(f"  run C == run B: {b == c}"); sys.exit(0 if b == c else 1)
EOF
done
gio trash $T 2>/dev/null
[ $fail = 0 ] && echo "Q8CACHE TESTS PASSED" || { echo "Q8CACHE TESTS FAILED"; exit 1; }
