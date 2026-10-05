#!/usr/bin/env python
"""
Score submission file(s) against a labelled test set using the competition metric (macro F1).

    python score.py submission.csv
    python score.py submission.csv --truth test_labeled.csv
    python score.py sub_a.csv sub_b.csv sub_c.csv        # compare several

Column names are auto-detected:
    submission : id + one of y_pred, prediction, pred, label, y, target
    truth      : id + one of y, label, y_true, target, class

Beyond the headline score it reports a bootstrap confidence interval and, for multiple files,
whether the gaps between them are larger than resampling noise. A 0.003 difference on 11,955 rows
usually is not, and knowing that is the difference between a real improvement and a coin flip.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

CLASS_NAMES = {0: "Explicitly Toxic", 1: "Subtly Toxic", 2: "Non-toxic"}
PRED_ALIASES = ["y_pred", "prediction", "pred", "predicted", "label", "y", "target", "class"]
TRUE_ALIASES = ["y", "label", "y_true", "target", "class", "true"]
TRUTH_GUESSES = ["test_labeled.csv", "test_labelled.csv", "data/test_labeled.csv",
                 "her_will_main_test_labeled.csv"]


def pick_column(df, aliases, path, kind):
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    sys.exit(f"ERROR: no {kind} column in {path}. Looked for {aliases}, found {list(df.columns)}")


def load(path, aliases, kind):
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: {p} not found")
    df = pd.read_csv(p)
    lower = {c.lower(): c for c in df.columns}
    if "id" not in lower:
        sys.exit(f"ERROR: no `id` column in {p}. Found {list(df.columns)}")
    col = pick_column(df, aliases, p, kind)
    out = df[[lower["id"], col]].copy()
    out.columns = ["id", kind]
    return out, col


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def bootstrap_ci(y_true, y_pred, n=2000, seed=0, alpha=0.05):
    """Resample rows with replacement to get a CI on macro F1."""
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        vals[i] = macro_f1(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float(vals.std())


def paired_bootstrap(y_true, pred_a, pred_b, n=2000, seed=0):
    """P(model A scores >= model B) under row resampling. ~0.5 means indistinguishable."""
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    wins, diffs = 0, np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        da = macro_f1(y_true[idx], pred_a[idx])
        db = macro_f1(y_true[idx], pred_b[idx])
        diffs[i] = da - db
        wins += da >= db
    return wins / n, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions", nargs="+", help="one or more submission CSVs")
    ap.add_argument("--truth", default=None, help="labelled test CSV (auto-detected if omitted)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 to skip)")
    ap.add_argument("--public-frac", type=float, default=0.0,
                    help="if set, also simulate a public/private split of this fraction")
    ap.add_argument("--quiet", action="store_true", help="scores only, no per-class report")
    args = ap.parse_args()

    truth_path = args.truth or next((g for g in TRUTH_GUESSES if Path(g).exists()), None)
    if truth_path is None:
        sys.exit("ERROR: no truth file. Pass --truth <file>. Tried: " + ", ".join(TRUTH_GUESSES))

    truth, tcol = load(truth_path, TRUE_ALIASES, "truth")
    print(f"truth : {truth_path}  ({len(truth)} rows, label column '{tcol}')")
    if truth.truth.isna().any():
        sys.exit(f"ERROR: {truth.truth.isna().sum()} missing labels in the truth file")
    bad = set(truth.truth.unique()) - {0, 1, 2}
    if bad:
        sys.exit(f"ERROR: truth labels outside 0/1/2: {sorted(bad)}")
    if truth.id.duplicated().any():
        sys.exit(f"ERROR: duplicate ids in the truth file ({truth.id.duplicated().sum()})")

    results = []
    for sub_path in args.submissions:
        sub, pcol = load(sub_path, PRED_ALIASES, "pred")
        name = Path(sub_path).name

        # --- validation -------------------------------------------------------------
        problems = []
        if len(sub) != len(truth):
            problems.append(f"row count {len(sub)} != {len(truth)}")
        if sub.id.duplicated().any():
            problems.append(f"{sub.id.duplicated().sum()} duplicate ids")
        missing = set(truth.id) - set(sub.id)
        extra = set(sub.id) - set(truth.id)
        if missing:
            problems.append(f"{len(missing)} ids missing")
        if extra:
            problems.append(f"{len(extra)} unexpected ids")
        bad = set(pd.unique(sub.pred.dropna())) - {0, 1, 2}
        if bad:
            problems.append(f"predictions outside 0/1/2: {sorted(bad)[:5]}")
        if sub.pred.isna().any():
            problems.append(f"{sub.pred.isna().sum()} missing predictions")
        if problems:
            print(f"\n{name}: INVALID -> " + "; ".join(problems))
            continue

        m = truth.merge(sub, on="id", how="left")
        yt, yp = m.truth.values.astype(int), m.pred.values.astype(int)
        score = macro_f1(yt, yp)
        results.append({"file": name, "macro_f1": score, "yt": yt, "yp": yp,
                        "pcol": pcol, "order_ok": bool((sub.id.values == truth.id.values).all())})

        print(f"\n{'='*70}\n{name}   (prediction column '{pcol}')\n{'='*70}")
        print(f"  MACRO F1 : {score:.5f}        <- competition metric")
        print(f"  accuracy : {(yt == yp).mean():.5f}")
        if not results[-1]["order_ok"]:
            print("  note: id order differs from the truth file; joined on id (Kaggle does the same)")

        if args.boot:
            lo, hi, sd = bootstrap_ci(yt, yp, n=args.boot)
            print(f"  95% CI   : [{lo:.5f}, {hi:.5f}]   sd {sd:.5f}   ({args.boot} resamples)")

        if args.public_frac > 0:
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(yt))
            k = int(len(yt) * args.public_frac)
            pub, prv = idx[:k], idx[k:]
            sp, sv = macro_f1(yt[pub], yp[pub]), macro_f1(yt[prv], yp[prv])
            print(f"  simulated public ({args.public_frac:.0%}) {sp:.5f} | "
                  f"private {sv:.5f} | gap {sp-sv:+.5f}")

        if not args.quiet:
            print()
            print(classification_report(yt, yp, digits=3, zero_division=0,
                                        target_names=[CLASS_NAMES[i] for i in range(3)]))
            cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
            print("  confusion matrix (rows = true):")
            print("   " + "\n   ".join(str(cm).split("\n")))
            n_err = cm.sum() - np.trace(cm)
            if n_err:
                print(f"  0<->1 confusion: {cm[0,1]+cm[1,0]} of {n_err} errors "
                      f"({(cm[0,1]+cm[1,0])/n_err:.1%})")
            print("  predicted mix:",
                  pd.Series(yp).value_counts(normalize=True).sort_index().round(4).to_dict())
            print("  true mix     :",
                  pd.Series(yt).value_counts(normalize=True).sort_index().round(4).to_dict())

    # --- comparison ---------------------------------------------------------------------
    if len(results) > 1:
        print(f"\n{'='*70}\nRANKING\n{'='*70}")
        results.sort(key=lambda r: -r["macro_f1"])
        for i, r in enumerate(results, 1):
            delta = r["macro_f1"] - results[0]["macro_f1"]
            print(f"  {i}. {r['file']:<40s} {r['macro_f1']:.5f}"
                  + (f"   ({delta:+.5f})" if i > 1 else "   <- best"))

        if args.boot:
            print(f"\n{'='*70}\nIS THE GAP REAL?  (paired bootstrap vs the best file)\n{'='*70}")
            best = results[0]
            for r in results[1:]:
                p, lo, hi = paired_bootstrap(best["yt"], best["yp"], r["yp"], n=args.boot)
                straddles = lo <= 0 <= hi
                verdict = ("INDISTINGUISHABLE - the CI on the difference includes 0"
                           if straddles else
                           "clearly better" if p > 0.95 else
                           "probably better" if p > 0.85 else "unclear")
                print(f"  {best['file']} vs {r['file']}")
                print(f"    P(best wins) = {p:.3f}   diff 95% CI [{lo:+.5f}, {hi:+.5f}]   {verdict}")
            print("\n  Read the CI on the DIFFERENCE, not the gap between the two scores.")
            print("  If it includes 0 the ordering is noise and could flip on the private split.")
            print("  Note this is a PAIRED test: it asks whether one file beats the other on")
            print("  these same rows, so two models that differ only on a handful of rows can")
            print("  still separate cleanly. It does not tell you either one generalises.")

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
