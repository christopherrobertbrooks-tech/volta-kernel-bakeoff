FIRST VERSION, WITH A BUG -- do not quote. It reused conversions across llama.cpp's mul_mat_id per-expert stack
temporaries (same address, same scratch buffer, different contents); perplexity 5.8296 vs 5.8300 showed it. Fixed version:
results/2026-10-20-fuse-q8-cache.
