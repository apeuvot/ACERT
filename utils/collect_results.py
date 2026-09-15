"""Gather the results of several runs into a single csv table.

Every run directory produced by ``run_downstream_custom_multiple_fold.py`` holds a
``results_<split>.txt`` file with a ``+++ SUMMARY for <split> +++`` block. This
script extracts that block for each run matching a glob and writes one row per run,
each metric formatted as ``mean ± std``.

This is how the per-context-length tables (Figure 1 of the paper) were produced:

    python utils/collect_results.py \\
        --results-dir results/IEMOCAP \\
        --pattern "hubert-large-ll60k_t=*_mean_pool__*" \\
        --out results_iemocap.csv
"""

import argparse
import csv
import glob
import os
import re

METRICS = ["UAR", "WAR", "macroF1", "weightedF1"]

# "Mean UAR [%]: 73.02", "Fold Std. UAR [%]: 3.46", "Run Median UAR [%]: 73.30"
_STAT_RE = re.compile(
    r"^(?P<stat>Mean|Fold Std\.|Fold Median|Run Std\.|Run Median)\s+"
    r"(?P<metric>\S+)\s+\[%\]:\s*(?P<value>-?[\d.]+)\s*$"
)


def parse_summary(path, split):
    """Return {metric: {"Mean": v, "Run Std.": v, ...}} for the given split."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    try:
        start = lines.index(f"+++ SUMMARY for {split} +++") + 1
    except ValueError:
        return None

    stats = {}
    for line in lines[start:]:
        # the SUMMARY block ends where the next block starts
        if line.startswith("+++"):
            break
        match = _STAT_RE.match(line.strip())
        if match:
            stats.setdefault(match["metric"], {})[match["stat"]] = float(match["value"])
    return stats or None


def get_context_duration(run_dir):
    """Context duration of a run, read from arguments.txt or from the directory name."""
    args_file = os.path.join(run_dir, "arguments.txt")
    if os.path.isfile(args_file):
        with open(args_file, encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "context_duration" and value.strip() != "None":
                    try:
                        return float(value.strip())
                    except ValueError:
                        pass
    match = re.search(r"t=(\d+(?:\.\d+)?)", os.path.basename(run_dir))
    return float(match.group(1)) if match else None


def format_cell(stats, metric, std_kind, decimals):
    """``73.02 ± 3.46`` (or just the mean if the requested std is missing)."""
    values = stats.get(metric)
    if not values or "Mean" not in values:
        return ""
    mean = f"{values['Mean']:.{decimals}f}"
    std = values.get(f"{std_kind} Std.")
    return mean if std is None else f"{mean} ± {std:.{decimals}f}"


def collect(results_dir, pattern, split, tmin, tmax):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, pattern))):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)

        duration = get_context_duration(run_dir)
        if duration is not None:
            if tmin is not None and duration < tmin:
                continue
            if tmax is not None and duration > tmax:
                continue

        results_file = os.path.join(run_dir, f"results_{split}.txt")
        if not os.path.isfile(results_file):
            print(f"[skip] no results_{split}.txt in {name}")
            continue

        stats = parse_summary(results_file, split)
        if stats is None:
            print(f"[skip] no 'SUMMARY for {split}' block in {name}")
            continue

        rows.append({"run": name, "t": duration, "stats": stats})

    rows.sort(key=lambda row: (row["t"] is None, row["t"], row["run"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="directory containing the run directories")
    parser.add_argument("--pattern", default="*", help="glob selecting the runs to gather")
    parser.add_argument("--split", default="Test", help="split to read (results_<split>.txt)")
    parser.add_argument("--std", default="Run", choices=["Fold", "Run"],
                        help="standard deviation shown in the cells: across runs (default) "
                             "or across folds")
    parser.add_argument("--tmin", type=float, default=None, help="minimum context duration (inclusive)")
    parser.add_argument("--tmax", type=float, default=None, help="maximum context duration (inclusive)")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--out", default=None, help="output csv path (default: print only)")
    args = parser.parse_args()

    rows = collect(args.results_dir, args.pattern, args.split, args.tmin, args.tmax)
    if not rows:
        parser.error(f"no run matching '{args.pattern}' in {args.results_dir}")

    table = [["t [s]", "run"] + METRICS]
    for row in rows:
        t = "" if row["t"] is None else f"{row['t']:g}"
        table.append([t, row["run"]]
                     + [format_cell(row["stats"], m, args.std, args.decimals) for m in METRICS])

    print("\n".join(" | ".join(line) for line in table))

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(table)
        print(f"[ok] csv written: {args.out}")


if __name__ == "__main__":
    main()
