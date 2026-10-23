"""Compare q8cache_test outputs, cache on against cache off, block by block (A1, A2, B, C: 2048 floats each).

Exit 1 if any block differs, or if the cache-off run C equals run B (the inputs differ, so equal outputs would mean
the test can't tell a stale reuse from a correct one)."""
import sys

NAMES = ["A1 (before the in-place op)", "A2 (after it)", "B (branches)", "C (2nd of 2 unsynced runs)"]
N = 2048 * 4


def blocks(path):
    with open(path, "rb") as f:
        d = f.read()
    return [d[i * N:(i + 1) * N] for i in range(len(NAMES))]


def main(on_path, off_path, label):
    on, off = blocks(on_path), blocks(off_path)
    ok = True
    for name, a, b in zip(NAMES, on, off):
        same = a == b
        ok &= same
        print(f"{label} {name}: cache on {'==' if same else 'DIFFERS FROM'} cache off")
    sees = off[2] != off[3]
    print(f"{label} instrument: cache-off run C differs from run B: {sees}")
    return 0 if ok and sees else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
