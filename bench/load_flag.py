#!/usr/bin/env python3
"""One line about a round from its 100 ms samples (gpu-load-<round>.csv).

While the GPU was busy, did the SM clock stay at the locked 1380 MHz, and was any
clock-event reason active besides idle (0x1) and the clock setting itself (0x2)?
Volatile uncorrected ECC must stay 0 and must not rise. A file with no usable busy
sample (empty, unreadable, or utilization never above 0) is flagged too.

    load_flag.py ROUND gpu-load-ROUND.csv

A clean round prints the same line as before. A bad round still ends with
"  ** FLAGGED **". Exit status stays 0; the line is the result (overnight.sh logs it
through a pipe). Wrong arguments exit 2.
"""
import sys

LOCKED_MHZ = 1380
# clocks_event_reasons.active: bits other than these flag a sample that is not known-idle.
#   0x1 GpuIdle, 0x2 ApplicationsClocksSetting (the -lgc lock)
#   0x4 SwPowerCap, 0x8 HwSlowdown, 0x10 SyncBoost, 0x20 SwThermalSlowdown,
#   0x40 HwThermalSlowdown, 0x80 HwPowerBrakeSlowdown, 0x100 DisplayClockSetting
OK_REASONS = 0x1 | 0x2
# nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,
# utilization.gpu,clocks_event_reasons.active,ecc.errors.uncorrected.volatile.total
SM, TEMP, UTIL, REASON, ECC = 1, 4, 5, 6, 7
COLUMNS = 8


def leading_int(text):
    """Integer at the start of an nvidia-smi field ('1380 MHz', '100 %', '46')."""
    parts = text.strip().split()
    if not parts:
        raise ValueError("empty")
    word = parts[0]
    for suffix in ("MHz", "Mhz", "W", "C", "%"):
        if word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return int(word)


def parse_int(text):
    try:
        return leading_int(text)
    except (ValueError, IndexError):
        return None


def parse_reason(text):
    parts = text.strip().split()
    if not parts:
        return None
    try:
        return int(parts[0], 16)
    except ValueError:
        return None


def split_samples(text):
    """Complete csv rows, and how many newline-terminated rows were too short to use.

    sample_stop kills nvidia-smi, so the file can end in a partial line with no
    newline. That tail is dropped: a good run must not be flagged for it. A partial
    file whose only line was cut off counts as malformed (there is nothing else to judge).
    """
    malformed = 0
    if text and not text.endswith("\n"):
        head, nl, _tail = text.rpartition("\n")
        if not nl:
            return [], int(bool(text.strip()))
        text = head + "\n"
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < COLUMNS:
            malformed += 1
            continue
        rows.append(fields)
    return rows, malformed


def plural(n, phrase):
    return f"{n} {phrase}" if n == 1 else f"{n} {phrase}s"


def judge(text):
    rows, malformed = split_samples(text)
    busy = low = odd = 0
    minclk = None
    maxt = 0
    reasons = 0
    clock_bad = reason_bad = util_bad = ecc_bad = 0
    ecc_vals = []
    for fields in rows:
        clk = parse_int(fields[SM])
        temp = parse_int(fields[TEMP])
        util = parse_int(fields[UTIL])
        why = parse_reason(fields[REASON])
        ecc = parse_int(fields[ECC])
        if ecc is None:
            ecc_bad += 1
        else:
            ecc_vals.append(ecc)
        known_busy = util is not None and util > 0
        # util == 0 is idle: the card is expected to clock down and report GpuIdle.
        # util is None is not idle — a low clock or throttle bit on that row still counts.
        known_idle = util == 0
        if known_busy:
            busy += 1
            if temp is not None:
                maxt = max(maxt, temp)
        elif util is None:
            util_bad += 1
        if known_idle:
            continue
        if clk is None:
            if known_busy:
                clock_bad += 1
        else:
            minclk = clk if minclk is None else min(minclk, clk)
            if clk < LOCKED_MHZ:
                low += 1
        if why is None:
            if known_busy:
                reason_bad += 1
        elif why & ~OK_REASONS:
            odd += 1
            reasons |= why & ~OK_REASONS
    notes = []
    if busy == 0:
        notes.append("no usable busy samples")
    if util_bad:
        notes.append(f"unreadable utilization on {plural(util_bad, 'sample')}")
    if clock_bad:
        notes.append(f"unreadable SM clock on {plural(clock_bad, 'busy sample')}")
    if reason_bad:
        notes.append(f"unreadable clock-event reasons on {plural(reason_bad, 'busy sample')}")
    if malformed:
        notes.append(plural(malformed, "malformed sample"))
    if ecc_bad and not ecc_vals:
        notes.append("ecc uncorrected unreadable")
    elif ecc_bad:
        notes.append(f"ecc uncorrected unreadable on {plural(ecc_bad, 'sample')}")
    if ecc_vals and (any(v != 0 for v in ecc_vals) or max(ecc_vals) > ecc_vals[0]):
        peak = max(ecc_vals)
        if peak > ecc_vals[0]:
            notes.append(f"ecc uncorrected {ecc_vals[0]}->{peak}")
        else:
            notes.append(f"ecc uncorrected {peak}")
    return busy, minclk, maxt, low, odd, reasons, notes


def format_line(round_id, busy, minclk, maxt, low, odd, reasons, notes):
    clk = "-" if minclk is None else str(minclk)
    reason = f" (0x{reasons:x})" if reasons else ""
    extra = (", " + ", ".join(notes)) if notes else ""
    mark = "  ** FLAGGED **" if (low or odd or notes) else ""
    return (
        f"round {round_id} under load: {busy} busy samples, SM clock min {clk} MHz, "
        f"max {maxt} C, {low} below 1380, {odd} with other clock-event reasons"
        f"{reason}{extra}{mark}"
    )


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: load_flag.py ROUND gpu-load-ROUND.csv\n")
        return 2
    round_id, path = argv[1], argv[2]
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as err:
        why = err.strerror or "unknown error"
        print(format_line(round_id, 0, None, 0, 0, 0, 0, [f"could not read samples ({why})"]))
        return 0
    print(format_line(round_id, *judge(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
