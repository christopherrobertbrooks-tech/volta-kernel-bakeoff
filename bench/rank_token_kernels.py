# Issue #3 step 1: rank llama.cpp's per-token kernels and their back-to-back pairs from an nsys sqlite export
import collections
import itertools
import re
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1]); ntok = int(sys.argv[2])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, shortName, demangledName, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
def short(r):
    n = names[r[2]]; d = names.get(r[3], n)
    m = re.match(r"(?:void )?([A-Za-z_0-9:]+)(<[^(]*>)?", d)
    base = m.group(1) if m else n
    t = m.group(2) or "" if m else ""
    if base in ("mul_mat_vec_q", "mul_mat_vec_f", "rms_norm_f32", "k_bin_bcast", "unary_op_kernel", "unary_gated_op_kernel",
                "quantize_q8_1", "cpy_scalar", "k_get_rows_float", "k_set_rows"):
        # keep the first template argument: the quant type / op
        a = t.strip("<>").split(",")[0].strip() if t else ""
        return f"{base}<{a}>"
    return base
tot = collections.defaultdict(float); cnt = collections.Counter()
seq = []
for r in rows:
    k = short(r); dt = (r[1] - r[0]) / 1e3
    tot[k] += dt; cnt[k] += 1; seq.append((k, dt, r[0], r[1]))
busy = sum(tot.values())
print(f"{len(rows)} kernels, {ntok} tokens: {len(rows)/ntok:.0f} per token, {busy/ntok/1e3:.3f} ms busy per token")
print(f"\n{'kernel':48s} {'per tok':>7s} {'us/tok':>8s} {'avg us':>7s} {'share':>6s}")
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:30]:
    print(f"{k[:48]:48s} {cnt[k]/ntok:7.1f} {v/ntok:8.1f} {v/cnt[k]:7.2f} {100*v/busy:5.1f}%")
# back-to-back pairs of kernels under 10 us (fusion candidates): count, combined time, and the gap between them
pairs = collections.defaultdict(lambda: [0, 0.0, 0.0])
for (a, da, sa, ea), (b, db_, sb, eb) in itertools.pairwise(seq):
    if da < 10 and db_ < 10:
        p = pairs[(a, b)]; p[0] += 1; p[1] += da + db_; p[2] += max(0, (sb - ea) / 1e3)
print("\nback-to-back pairs of small (<10 us) kernels, by combined time per token")
print(f"{'first':40s} {'then':40s} {'per tok':>7s} {'us/tok':>7s} {'gap us/tok':>10s}")
for (a, b), (n, t, g) in sorted(pairs.items(), key=lambda x: -x[1][1])[:25]:
    print(f"{a[:40]:40s} {b[:40]:40s} {n/ntok:7.1f} {t/ntok:7.1f} {g/ntok:10.1f}")
