"""Summarise a bench/fuse-ab.sh run: per side medians, and the per-round paired ratio (on / off in the same round),
which cancels slow drift in the machine's speed. Optional: a time window (HH:MM-HH:MM) to report rounds inside and
outside it separately (e.g. while another job was running on the box), or a round range (r14-30)."""
import csv
import glob
import re
import statistics
import sys


def main(d, window=None):
    times = {}
    with open(f"{d}/run.log") as fh:
        for line in fh:
            m = re.match(r"\S+ (\d\d:\d\d):\d\d round (\d+) (on|off): ([\d.]+) tok/s", line)
            if m:
                times[(int(m.group(2)), m.group(3))] = m.group(1)
    tok = {}
    for f in glob.glob(f"{d}/bench-*-*.csv"):
        side, r = re.search(r"bench-(on|off)-(\d+)\.csv", f).groups()
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if rows:   # a run still in progress has an empty file
            tok[(int(r), side)] = float(rows[-1]["avg_ts"])
    rounds = sorted({r for r, _ in tok if (r, "on") in tok and (r, "off") in tok})

    def inside(r):
        if not window:
            return False
        if window.startswith("r"):
            a, b = window[1:].split("-")
            return int(a) <= r <= int(b)
        a, b = window.split("-")
        return any(a <= times.get((r, s), "99:99") <= b for s in ("on", "off"))

    groups = {"all": rounds}
    if window:
        groups[f"outside {window}"] = [r for r in rounds if not inside(r)]
        groups[f"inside {window}"] = [r for r in rounds if inside(r)]
    for name, rs in groups.items():
        if not rs:
            continue
        on = [tok[(r, "on")] for r in rs]
        off = [tok[(r, "off")] for r in rs]
        ratio = [tok[(r, "on")] / tok[(r, "off")] for r in rs]
        wins = sum(x > 1 for x in ratio)
        print(f"{name}: {len(rs)} rounds | off median {statistics.median(off):.2f} | on median {statistics.median(on):.2f} | "
              f"paired gain median {100 * (statistics.median(ratio) - 1):+.2f}% (min {100 * (min(ratio) - 1):+.2f}%, "
              f"max {100 * (max(ratio) - 1):+.2f}%) | on faster in {wins}/{len(rs)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
