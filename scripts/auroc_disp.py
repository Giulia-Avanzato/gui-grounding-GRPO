import json, sys, os
import numpy as np

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def auroc(score, err):
    pos, neg = score[err], score[~err]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size

def boot_ci(score, err, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n, vals = len(score), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        a = auroc(score[i], err[i])
        if not np.isnan(a):
            vals.append(a)
    return np.percentile(vals, [2.5, 97.5])

def report(name, rows):
    rows = [r for r in rows if r.get("dispersion") is not None and np.isfinite(r["dispersion"])]
    if len(rows) < 40:
        return
    s = np.array([r["dispersion"] for r in rows], float)
    e = np.array([not r["hit"] for r in rows])
    if e.sum() < 10 or (~e).sum() < 10:
        print(f"{name:38s} n={len(rows):4d}  troppo pochi errori/successi, salto")
        return
    a = auroc(s, e)
    lo, hi = boot_ci(s, e)
    print(f"{name:38s} n={len(rows):4d} err={int(e.sum()):4d}  "
          f"AUROC={a:.3f} [{lo:.3f},{hi:.3f}]  "
          f"disp media: giuste={s[~e].mean():.3f} sbagliate={s[e].mean():.3f}  "
          f"disp=0: {(s == 0).mean():.0%}")

for p in sys.argv[1:]:
    rows = load(p)
    base = os.path.basename(p)
    report(base, rows)
    for t in sorted({r.get("data_type") for r in rows}):
        report(f"   {base} [{t}]", [r for r in rows if r.get("data_type") == t])
