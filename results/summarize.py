# Medians over every round of a results folder: sweep-*.csv (ours + cuBLAS) and llama-*.csv (llama.cpp).
#   python3 results/summarize.py results/2026-10-19 > results/2026-10-19/medians.csv
import csv
import glob
import statistics
import sys
from collections import defaultdict

d = sys.argv[1]
us = defaultdict(list)
for f in sorted(glob.glob(f"{d}/sweep-*.csv") + glob.glob(f"{d}/llama-*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            us[(r["contender"], r["shape"], int(r["M"]))].append(float(r["us"]))
print("contender,shape,M,median_us,min_us,max_us,rounds")
for (c, s, m), v in sorted(us.items()):
    print(f"{c},{s},{m},{statistics.median(v):.2f},{min(v):.2f},{max(v):.2f},{len(v)}")
