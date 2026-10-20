#!/usr/bin/env python3
"""One line about a round from its 100 ms samples (gpu-load-<round>.csv): while the GPU was busy, did the SM clock stay
at the locked 1380 MHz, and was any clock-event reason active besides idle (0x1) and the clock setting itself (0x2)?

    load_flag.py ROUND gpu-load-ROUND.csv
"""
import sys
r, path = sys.argv[1], sys.argv[2]
busy = low = odd = 0; minclk = None; maxt = 0; reasons = 0
for line in open(path):
    f = [x.strip() for x in line.split(",")]
    if len(f) < 8: continue
    try:
        clk, temp, util, why = int(f[1].split()[0]), int(f[4]), int(f[5].split()[0]), int(f[6], 16)
    except ValueError:
        continue
    if util == 0: continue
    busy += 1; maxt = max(maxt, temp); minclk = clk if minclk is None else min(minclk, clk)
    low += clk < 1380
    if why & ~0x3: odd += 1; reasons |= why & ~0x3
flag = "  ** FLAGGED **" if (low or odd) else ""
print(f"round {r} under load: {busy} busy samples, SM clock min {minclk if busy else '-'} MHz, max {maxt} C, "
      f"{low} below 1380, {odd} with other clock-event reasons{f' (0x{reasons:x})' if reasons else ''}{flag}")
