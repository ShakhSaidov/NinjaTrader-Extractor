"""
NQ Parquet Verifier: OLD (nq_1min.parquet) vs NEW (NQ_continuous_raw.parquet)

WHAT THIS DOES
---------------
1. Loads both files, normalizing OLD (datetime index) and NEW (datetime
   column + contract column) into a common shape.
2. Reports internal completeness for EACH file separately (missing minute
   bars), so "is NEW missing anything" can be answered independent of OLD.
3. Compares every overlapping timestamp for OHLC + volume differences.
4. Aggregates per-day, ranks worst-first, and flags EVERY date range above
   tolerance -- not just the one you already spotted (Oct 16-20, 2020).
5. Computes an "offset_consistency" score per day: values near 1.0 mean the
   diff was roughly CONSTANT across the whole day, which is the fingerprint
   of a back-adjustment/roll-offset difference rather than random data
   corruption. Low or negative values mean the diff is erratic within the
   day -- treat those as higher priority for manual/visual review.
6. Lists each contract's active date range in NEW, so you can see roll
   boundaries and check whether a flagged date sits right on one.

USAGE
-----
    python verify_nq_parquets.py \
        --old ./nq_1min.parquet \
        --new ./NQ_1min_output/continuous/NQ_continuous_raw.parquet \
        --out ./nq_verify_report \
        --price-tol 1.0

OUTPUTS (written to --out)
---------------------------
    summary.txt                   human-readable top-line report
    daily_diff_summary.csv        per-day diff stats, worst day first
    mismatched_bars_detail.csv    every bar exceeding --price-tol, full OHLCV both sides
    missing_from_new.csv          timestamps in OLD but absent from NEW
    missing_from_old.csv          timestamps in NEW but absent from OLD
    session_gaps_old.csv          missing minute bars inside OLD's own timeline
    session_gaps_new.csv          missing minute bars inside NEW's own timeline
    contract_segments_new.csv     each contract's date range + row count in NEW
    duplicate_datetimes_new.csv   duplicate timestamps in NEW (roll overlap check)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_old(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    dupes = df.index.duplicated().sum()
    if dupes:
        print(f"[WARN] OLD has {dupes} duplicate datetime rows. Keeping first.")
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def load_new(path: str):
    raw = pd.read_parquet(path)
    # Normalize dtype explicitly -- this also happens to be the exact fix
    # for the "1970" chart bug if you reuse this pattern in chart code.
    raw["datetime"] = pd.to_datetime(raw["datetime"]).astype("datetime64[ns]")
    raw = raw.sort_values("datetime")

    dupe_mask = raw["datetime"].duplicated(keep=False)
    dupes_df = raw[dupe_mask].copy()

    dupes = raw["datetime"].duplicated().sum()
    if dupes:
        print(f"[WARN] NEW has {dupes} duplicate datetime rows (likely contract-roll overlap). Keeping first.")

    df = raw.drop_duplicates(subset="datetime", keep="first").set_index("datetime")
    return df, dupes_df, raw


def find_session_gaps(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Flags every missing minute-bar timestamp in [min, max] of this file's
    own index. NQ trades close to 24h with a daily maintenance break, so
    expect a recurring daily gap -- the point is to eyeball whether gaps
    are ONLY that expected break, or whether there are extra unexplained
    holes.
    """
    idx = df.index
    full_range = pd.date_range(idx.min(), idx.max(), freq=freq)
    missing = full_range.difference(idx)
    gap_df = pd.DataFrame({"missing_datetime": missing})
    if len(gap_df):
        gap_df["date"] = gap_df["missing_datetime"].dt.date
        gap_df["time"] = gap_df["missing_datetime"].dt.time
    return gap_df


def contract_segments(new_raw: pd.DataFrame) -> pd.DataFrame:
    if "contract" not in new_raw.columns:
        return pd.DataFrame()
    g = new_raw.reset_index().groupby("contract")["datetime"]
    seg = g.agg(["min", "max", "count"]).rename(
        columns={"min": "first_datetime", "max": "last_datetime", "count": "rows"}
    )
    return seg.sort_values("first_datetime")


def compare(old: pd.DataFrame, new: pd.DataFrame, price_tol_points: float):
    common_idx = old.index.intersection(new.index)
    only_old = old.index.difference(new.index)
    only_new = new.index.difference(old.index)

    o = old.loc[common_idx, ["open", "high", "low", "close", "volume"]].sort_index()
    n = new.loc[common_idx, ["open", "high", "low", "close", "volume"]].sort_index()

    diff = pd.DataFrame(index=common_idx)
    for col in ["open", "high", "low", "close"]:
        diff[f"{col}_old"] = o[col]
        diff[f"{col}_new"] = n[col]
        diff[f"{col}_diff"] = n[col] - o[col]
    diff["volume_old"] = o["volume"]
    diff["volume_new"] = n["volume"]
    diff["volume_diff"] = n["volume"] - o["volume"]
    diff["max_abs_price_diff"] = diff[["open_diff", "high_diff", "low_diff", "close_diff"]].abs().max(axis=1)

    mismatched = diff[diff["max_abs_price_diff"] > price_tol_points].copy()
    return diff, mismatched, only_old, only_new


def daily_summary(diff: pd.DataFrame) -> pd.DataFrame:
    d = diff.copy()
    d["date"] = d.index.date
    g = d.groupby("date")["max_abs_price_diff"]
    summary = g.agg(["count", "mean", "max", "std"]).rename(
        columns={"count": "bars_compared", "mean": "mean_abs_diff", "max": "max_abs_diff", "std": "std_abs_diff"}
    )
    summary["offset_consistency"] = 1 - (summary["std_abs_diff"] / summary["mean_abs_diff"].replace(0, np.nan))
    return summary.sort_values("max_abs_diff", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="./nq_verify_report")
    ap.add_argument("--price-tol", type=float, default=1.0,
                     help="points; abs diff above this on any O/H/L/C flags the bar")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OLD: {args.old}")
    old = load_old(args.old)
    print(f"  rows={len(old):,}  range=[{old.index.min()} .. {old.index.max()}]")

    print(f"Loading NEW: {args.new}")
    new, new_dupes, new_raw = load_new(args.new)
    print(f"  rows={len(new):,}  range=[{new.index.min()} .. {new.index.max()}]")

    diff, mismatched, only_old, only_new = compare(old, new, args.price_tol)

    print(f"\nCommon timestamps: {len(diff):,}")
    print(f"Missing from NEW (present in OLD only): {len(only_old):,}")
    print(f"Missing from OLD (present in NEW only): {len(only_new):,}")
    print(f"Bars exceeding {args.price_tol} pt tolerance: {len(mismatched):,} "
          f"({100 * len(mismatched) / max(len(diff), 1):.4f}%)")

    daily = daily_summary(diff)
    daily.to_csv(outdir / "daily_diff_summary.csv")
    mismatched.to_csv(outdir / "mismatched_bars_detail.csv")
    pd.Series(only_old, name="datetime").to_csv(outdir / "missing_from_new.csv", index=False)
    pd.Series(only_new, name="datetime").to_csv(outdir / "missing_from_old.csv", index=False)

    find_session_gaps(old).to_csv(outdir / "session_gaps_old.csv", index=False)
    find_session_gaps(new).to_csv(outdir / "session_gaps_new.csv", index=False)

    segs = contract_segments(new_raw)
    if not segs.empty:
        segs.to_csv(outdir / "contract_segments_new.csv")

    new_dupes.to_csv(outdir / "duplicate_datetimes_new.csv", index=False)

    worst = daily.head(30)
    with open(outdir / "summary.txt", "w") as f:
        f.write("NQ PARQUET VERIFIER SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"OLD: {args.old}\n  rows={len(old):,}  range=[{old.index.min()} .. {old.index.max()}]\n\n")
        f.write(f"NEW: {args.new}\n  rows={len(new):,}  range=[{new.index.min()} .. {new.index.max()}]\n\n")
        f.write(f"Common timestamps: {len(diff):,}\n")
        f.write(f"Missing from NEW (present in OLD only): {len(only_old):,}\n")
        f.write(f"Missing from OLD (present in NEW only): {len(only_new):,}\n")
        f.write(f"Bars exceeding {args.price_tol} pt tolerance: {len(mismatched):,} "
                f"({100 * len(mismatched) / max(len(diff), 1):.4f}%)\n\n")
        f.write("TOP 30 WORST DAYS BY MAX ABS PRICE DIFF:\n")
        f.write(worst.to_string())
        f.write("\n\nHOW TO READ 'offset_consistency':\n")
        f.write("Close to 1.0  -> diff was roughly CONSTANT all day (fingerprint of a\n")
        f.write("                 back-adjustment / contract-roll offset, not corruption).\n")
        f.write("Low/negative  -> diff jumps around within the day. Treat as HIGH priority\n")
        f.write("                 for manual visual review -- this shape does not look like\n")
        f.write("                 a clean adjustment offset.\n")

    print(f"\nReport written to: {outdir.resolve()}")
    print("Open summary.txt first, then daily_diff_summary.csv (already sorted worst-first).")


if __name__ == "__main__":
    main()