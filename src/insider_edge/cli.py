"""Command-line entry point.

Subcommands map to the plan's phases. Only the scaffolding-complete commands do
real work today; the rest announce the phase that will implement them.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import polars as pl

from insider_edge import config
from insider_edge.ingest import pipeline
from insider_edge.sources import edgar_bulk


def _cmd_download(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    quarters = edgar_bulk.quarters_range(args.start_year, args.start_q, args.end_year, args.end_q)
    print(f"Downloading {len(quarters)} quarter(s) into {config.RAW_DIR} ...")
    paths = asyncio.run(edgar_bulk.download_quarters(quarters))
    for p in paths:
        dest = edgar_bulk.extract_zip(p)
        print(f"  {p.name} -> {dest}")


def _cmd_parse(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    doc_types = ("4", "4/A") if args.include_amendments else ("4",)
    print(f"Parsing buys from {config.RAW_DIR} (document_types={doc_types}) ...")
    events = pipeline.build_events(document_types=doc_types, scan_footnotes=args.scan_footnotes)
    out = config.PARQUET_DIR / "events.parquet"
    print(f"Wrote {events.height:,} qualifying events -> {out}\n")

    report = pipeline.sanity_report(events)
    print("=== Phase 1 sanity report ===")
    print(json.dumps(report, indent=2, default=str))

    print("\n=== Spot-check rows: largest reliable-price buys (verify against EDGAR) ===")
    # Exclude footnoted/placeholder prices so the spot-check shows real, checkable buys.
    reliable = events.filter(
        (~pl.col("price_footnoted")) & (pl.col("price") > 0) & (pl.col("price") < 10_000)
    )
    sample = reliable.sort("value", descending=True, nulls_last=True).head(args.sample)
    for r in sample.to_dicts():
        url = pipeline.edgar_filing_url(r["accession"], r["issuer_cik"])
        print(
            f"  {r['ticker'] or '?':<6} {r['filing_date']}  "
            f"{(r['insider_name'] or '?')[:24]:<24} "
            f"{r['shares']:>12,.0f} @ {r['price']}  {url}"
        )


def _cmd_horizon(args: argparse.Namespace) -> None:
    from insider_edge.analysis import phase2

    events = pl.read_parquet(config.PARQUET_DIR / "events.parquet")
    print(
        f"Phase 2 horizon curve on {events.height:,} events "
        f"(sample={args.sample}, provider=yfinance SMOKE TEST) ...\n"
    )
    sample = None if args.sample == 0 else args.sample
    res = phase2.run_horizon(events, sample=sample, seed=args.seed, n_boot=args.n_boot)

    print(
        f"events used: {res['n_events']:,}   priced: {res['n_priced']:,}   "
        f"coverage: {res['coverage']}"
    )
    print("\n=== Population horizon curve (abnormal return vs SPY, 95% bootstrap CI) ===")
    for r in res["curve"].to_dicts():
        if r["mean_abn"] is None:
            print(f"  {r['horizon']:>3}: no data")
            continue
        if r["ci_lo"] > 0:
            flag = "   ** positive (CI clears 0)"
        elif r["ci_hi"] < 0:
            flag = "   xx negative (CI clears 0)"
        else:
            flag = "   ~ CI includes 0"
        print(
            f"  {r['horizon']:>3}: n={r['n']:>5}  mean={r['mean_abn']:+.4f}  "
            f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{flag}"
        )
    print(
        "\n⚠️  SMOKE TEST on yfinance: delisted names are absent, so this curve is "
        "survivorship-biased UPWARD and is NOT the verdict (plan §4.5)."
    )


def _todo(phase: str):
    def run(_: argparse.Namespace) -> None:
        raise SystemExit(f"Not implemented yet — {phase}.")

    return run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="insider-edge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Download + extract quarterly Form 345 sets (Phase 1a).")
    d.add_argument("--start-year", type=int, required=True)
    d.add_argument("--start-q", type=int, default=1, choices=[1, 2, 3, 4])
    d.add_argument("--end-year", type=int, required=True)
    d.add_argument("--end-q", type=int, default=4, choices=[1, 2, 3, 4])
    d.set_defaults(func=_cmd_download)

    pr = sub.add_parser("parse", help="Flatten TSVs -> events.parquet + sanity (Phase 1).")
    pr.add_argument("--include-amendments", action="store_true", help="Also include 4/A filings.")
    pr.add_argument("--scan-footnotes", action="store_true", help="Pre-2023 10b5-1 footnote scan.")
    pr.add_argument("--sample", type=int, default=8, help="Spot-check rows to print.")
    pr.set_defaults(func=_cmd_parse)
    hz = sub.add_parser("horizon", help="Population horizon curve (Phase 2 gate).")
    hz.add_argument("--sample", type=int, default=400, help="Events to sample (0 = all).")
    hz.add_argument("--seed", type=int, default=1)
    hz.add_argument("--n-boot", type=int, default=10_000)
    hz.set_defaults(func=_cmd_horizon)
    sub.add_parser("backtest", help="Dumb-basket backtest (Phase 4 gate).").set_defaults(
        func=_todo("Phase 4")
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
