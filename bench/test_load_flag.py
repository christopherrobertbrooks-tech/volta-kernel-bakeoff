#!/usr/bin/env python3
"""GPU-free checks for bench/load_flag.py.

    python3 bench/test_load_flag.py

Fixtures under bench/fixtures/ are synthetic nvidia-smi csv rows (the columns
overnight.sh asks for). Nothing here opens a device.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "load_flag.py"
FIX = ROOT / "fixtures"

checks = 0


def run(round_id, path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), round_id, str(path)],
        check=False, capture_output=True, text=True,
    )


def expect(round_id, path, line):
    global checks
    checks += 1
    proc = run(round_id, path)
    if proc.returncode != 0:
        raise SystemExit(f"{path}: exit {proc.returncode}\n{proc.stderr}")
    if proc.stderr:
        raise SystemExit(f"{path}: stderr not empty:\n{proc.stderr}")
    actual = proc.stdout
    if actual != line + "\n":
        raise SystemExit(f"{path}:\nexpected: {line!r}\nactual:   {actual!r}")


def expect_text(round_id, text, line):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "samples.csv"
        path.write_text(text)
        expect(round_id, path, line)


def main():
    expect(
        "llama-3", FIX / "good.csv",
        "round llama-3 under load: 4 busy samples, SM clock min 1380 MHz, max 46 C, "
        "0 below 1380, 0 with other clock-event reasons",
    )
    expect(
        "2", FIX / "clock_drop.csv",
        "round 2 under load: 3 busy samples, SM clock min 1230 MHz, max 56 C, "
        "2 below 1380, 0 with other clock-event reasons  ** FLAGGED **",
    )
    expect(
        "3", FIX / "throttle.csv",
        "round 3 under load: 3 busy samples, SM clock min 1380 MHz, max 61 C, "
        "0 below 1380, 2 with other clock-event reasons (0xc)  ** FLAGGED **",
    )
    expect(
        "6", FIX / "zero_empty.csv",
        "round 6 under load: 0 busy samples, SM clock min - MHz, max 0 C, "
        "0 below 1380, 0 with other clock-event reasons, no usable busy samples  ** FLAGGED **",
    )
    expect(
        "7", FIX / "zero_na.csv",
        "round 7 under load: 0 busy samples, SM clock min - MHz, max 0 C, "
        "0 below 1380, 0 with other clock-event reasons, no usable busy samples, "
        "unreadable utilization on 2 samples, ecc uncorrected unreadable  ** FLAGGED **",
    )
    expect(
        "err", FIX / "zero_error.csv",
        "round err under load: 0 busy samples, SM clock min - MHz, max 0 C, "
        "0 below 1380, 0 with other clock-event reasons, no usable busy samples, "
        "1 malformed sample  ** FLAGGED **",
    )
    expect(
        "8", FIX / "na_throttle.csv",
        "round 8 under load: 2 busy samples, SM clock min 1380 MHz, max 47 C, "
        "0 below 1380, 0 with other clock-event reasons, "
        "unreadable clock-event reasons on 2 busy samples  ** FLAGGED **",
    )
    expect(
        "9", FIX / "idle_only.csv",
        "round 9 under load: 0 busy samples, SM clock min - MHz, max 0 C, "
        "0 below 1380, 0 with other clock-event reasons, no usable busy samples  ** FLAGGED **",
    )
    expect(
        "4", FIX / "ecc_nonzero.csv",
        "round 4 under load: 2 busy samples, SM clock min 1380 MHz, max 45 C, "
        "0 below 1380, 0 with other clock-event reasons, ecc uncorrected 2  ** FLAGGED **",
    )
    expect(
        "5", FIX / "ecc_increase.csv",
        "round 5 under load: 2 busy samples, SM clock min 1380 MHz, max 46 C, "
        "0 below 1380, 0 with other clock-event reasons, ecc uncorrected 0->1  ** FLAGGED **",
    )
    expect(
        "e", FIX / "ecc_na.csv",
        "round e under load: 1 busy samples, SM clock min 1380 MHz, max 41 C, "
        "0 below 1380, 0 with other clock-event reasons, ecc uncorrected unreadable  ** FLAGGED **",
    )
    expect(
        "10", FIX / "temp_na_clock_drop.csv",
        "round 10 under load: 2 busy samples, SM clock min 1200 MHz, max 42 C, "
        "1 below 1380, 0 with other clock-event reasons  ** FLAGGED **",
    )
    expect(
        "u", FIX / "util_na_low_clock.csv",
        "round u under load: 2 busy samples, SM clock min 1200 MHz, max 45 C, "
        "1 below 1380, 1 with other clock-event reasons (0x4), "
        "unreadable utilization on 1 sample  ** FLAGGED **",
    )

    # A blank clock field used to raise IndexError and print nothing.
    expect_text(
        "blank",
        "2026/10/20 12:00:08.000, , 877 MHz, 40.00 W, 40, 100 %, 0x0000000000000002, 0\n",
        "round blank under load: 1 busy samples, SM clock min - MHz, max 40 C, "
        "0 below 1380, 0 with other clock-event reasons, "
        "unreadable SM clock on 1 busy sample  ** FLAGGED **",
    )
    # A blank utilization used to do the same, and hid the clock and the throttle bit.
    expect_text(
        "u2",
        "2026/10/20 12:00:08.100, 1100 MHz, 877 MHz, 40.00 W, 40, , 0x0000000000000004, 0\n",
        "round u2 under load: 0 busy samples, SM clock min 1100 MHz, max 0 C, "
        "1 below 1380, 1 with other clock-event reasons (0x4), "
        "no usable busy samples, unreadable utilization on 1 sample  ** FLAGGED **",
    )
    # Newline-only is an empty sample file. A short line between two good ones is not a tail.
    expect_text(
        "nl", "\n",
        "round nl under load: 0 busy samples, SM clock min - MHz, max 0 C, "
        "0 below 1380, 0 with other clock-event reasons, no usable busy samples  ** FLAGGED **",
    )
    good = (
        "2026/10/20 12:00:12.000, 1380 MHz, 877 MHz, 50.00 W, 41, 93 %, 0x0000000000000002, 0"
    )
    expect_text(
        "mid", good + "\nnot a sample\n" + good + "\n",
        "round mid under load: 2 busy samples, SM clock min 1380 MHz, max 41 C, "
        "0 below 1380, 0 with other clock-event reasons, 1 malformed sample  ** FLAGGED **",
    )
    # Killed nvidia-smi: the last line has no newline, even when it looks alarming.
    partial = (
        "2026/10/20 12:00:12.100, 1100 MHz, 877 MHz, 50.00 W, 70, 100 %, 0x0000000000000004, 9"
    )
    expect_text(
        "tail", good + "\n" + partial,
        "round tail under load: 1 busy samples, SM clock min 1380 MHz, max 41 C, "
        "0 below 1380, 0 with other clock-event reasons",
    )
    # The same bytes with a newline are a real sample: clock, throttle, and ECC all count.
    expect_text(
        "tail2", good + "\n" + partial + "\n",
        "round tail2 under load: 2 busy samples, SM clock min 1100 MHz, max 70 C, "
        "1 below 1380, 1 with other clock-event reasons (0x4), "
        "ecc uncorrected 0->9  ** FLAGGED **",
    )
    # The counter rose and came back; last == first must not hide the peak.
    expect_text(
        "peak",
        "2026/10/20 12:00:13.000, 1380 MHz, 877 MHz, 50.00 W, 41, 90 %, 0x0000000000000002, 0\n"
        "2026/10/20 12:00:13.100, 1380 MHz, 877 MHz, 50.00 W, 41, 90 %, 0x0000000000000002, 3\n"
        "2026/10/20 12:00:13.200, 1380 MHz, 877 MHz, 50.00 W, 41, 90 %, 0x0000000000000002, 0\n",
        "round peak under load: 3 busy samples, SM clock min 1380 MHz, max 41 C, "
        "0 below 1380, 0 with other clock-event reasons, ecc uncorrected 0->3  ** FLAGGED **",
    )
    # Nonzero without a rise (the series only falls) is still an ECC flag.
    expect_text(
        "fall",
        "2026/10/20 12:00:14.000, 1380 MHz, 877 MHz, 50.00 W, 41, 90 %, 0x0000000000000002, 4\n"
        "2026/10/20 12:00:14.100, 1380 MHz, 877 MHz, 50.00 W, 41, 90 %, 0x0000000000000002, 1\n",
        "round fall under load: 2 busy samples, SM clock min 1380 MHz, max 41 C, "
        "0 below 1380, 0 with other clock-event reasons, ecc uncorrected 4  ** FLAGGED **",
    )

    global checks
    checks += 1
    missing = run("missing", FIX / "no-such-file.csv")
    missing_line = (
        "round missing under load: 0 busy samples, SM clock min - MHz, max 0 C, "
        "0 below 1380, 0 with other clock-event reasons, "
        "could not read samples (No such file or directory)  ** FLAGGED **\n"
    )
    if missing.returncode != 0 or missing.stderr or missing.stdout != missing_line:
        raise SystemExit(
            f"missing file:\nstdout={missing.stdout!r}\nstderr={missing.stderr!r}\ncode={missing.returncode}"
        )

    checks += 1
    usage = subprocess.run([sys.executable, str(SCRIPT)], check=False, capture_output=True, text=True)
    if usage.returncode != 2 or usage.stdout or "usage:" not in usage.stderr:
        raise SystemExit(f"usage: code={usage.returncode} stdout={usage.stdout!r} stderr={usage.stderr!r}")

    print(f"ok, {checks} checks")


if __name__ == "__main__":
    main()
